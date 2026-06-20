"""
app/api/main.py
=================
Living Weather - FastAPI Giriş Noktası (HTTP API katmanı)

GÖREV:
  Şimdiye kadar yazılan 5 çekirdek ajan (Master, DI, LLM, AS, FL)
  sadece terminalden "python agent_master.py" ile çalıştırılabiliyordu
  - yani sadece bu makinedeki bir Python scripti olarak. Bu dosya,
  o ajan mantığının ÖNÜNE ince bir HTTP kabuğu koyar; böylece dışarıdan
  (curl, Postman, ileride bir frontend, Cloud Run üzerinden internet)
  bu sisteme erişilebilir hale gelir.

ÖNEMLİ:
  Bu dosya ajan mantığını DEĞİŞTİRMEZ. MasterAgent.process_request()
  zaten ne yapıyorsa onu yapmaya devam ediyor - bu dosya sadece onu
  "POST /weather" şeklinde dışarı açıyor.

NEDEN MasterAgent TEK SEFER OLUŞTURULUYOR (her istekte değil)?
  FL-Agent'ın tahmin geçmişi (_history) şu an bellek-içi (in-memory).
  Eğer her istekte yeni bir MasterAgent() oluştursak, FL-Agent'ın
  history'si her istekte sıfırlanır - yani "geçmiş tahminleri
  karşılaştırma" özelliği hiç çalışmaz. Bunun yerine, uygulama
  başlarken (startup) BİR KERE MasterAgent oluşturuyoruz ve tüm
  istekler aynı örneği (instance) paylaşıyor.

  NOT (STATUS.md'deki bilinen risk ile bağlantılı): Bu yaklaşım,
  uygulama TEK bir worker/process ile çalıştığı sürece doğru çalışır.
  Eğer ileride "uvicorn --workers 4" gibi birden fazla worker ile
  çalıştırırsak, her worker kendi MasterAgent'ını (ve kendi FL-Agent
  history'sini, kendi DI-Agent devre kesici durumunu) ayrı ayrı tutar
  - bu STATUS.md'de zaten not edilmiş bilinen bir sınırlama, Redis'e
  geçilince çözülecek.

ÇALIŞTIRMA (yerel makinede test için):
    cd C:\\Users\\levent.turgut\\living-weather
    uvicorn app.api.main:app --reload --port 8000

  Sonra tarayıcıda http://127.0.0.1:8000/docs adresine gidersen,
  FastAPI'nin otomatik ürettiği interaktif test arayüzünü görürsün
  (Swagger UI) - hiçbir ekstra kod yazmadan endpoint'leri deneyebilirsin.

Kaynak: Living Weather mimarisi (kullanıcı tasarımı), Day 5 whitepaper
        prensiplerinin HTTP katmanına taşınması.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.models import WeatherContext
from app.agents.master.agent import MasterAgent


app = FastAPI(
    title="Living Weather API",
    description=(
        "Kaggle 5-Day AI Agents Intensive - Bitirme Projesi. "
        "Multi-agent hava durumu sistemi: DI-Agent (gerçek veri), "
        "LLM-Agent (mevsimsel fallback), AS-Agent (uyarı/güvenlik), "
        "FL-Agent (geri bildirim/öğrenme), Master-Agent (orkestrasyon)."
    ),
    version="0.1.0",
)

# MasterAgent BİR KERE oluşturulur, tüm istekler arasında paylaşılır.
# (Yukarıdaki docstring'deki "NEDEN" açıklamasına bak.)
master_agent = MasterAgent()


# ----------------------------------------------------------------------
# REQUEST / RESPONSE MODELLERİ
# ----------------------------------------------------------------------
class WeatherRequest(BaseModel):
    """POST /weather için gövde (body) şeması."""
    location: str


class RouteRequest(BaseModel):
    """POST /route için gövde (body) şeması."""
    origin: str
    destination: str


class RouteResponse(BaseModel):
    """
    POST /route için yanıt şeması. RI-Agent'ın AgentResponse.payload'ı
    zaten bir dict olduğu için, burada AÇIK bir şema tanımlıyoruz
    (WeatherContext'te yaptığımız gibi var olan bir modeli yeniden
    kullanamadık çünkü RI-Agent'ın çıktısı WeatherContext'ten farklı
    bir şekle sahip - origin/destination/route_text/judge sonucu).
    """
    origin: str
    destination: str
    route_text: str
    text_source: str
    judge_score: float
    judge_passed: bool
    judge_reason: str
    confidence_score: float


# ----------------------------------------------------------------------
# ENDPOINT'LER
# ----------------------------------------------------------------------
@app.get("/health")
def health_check() -> dict:
    """
    Cloud Run / yük dengeleyici (load balancer) gibi sistemlerin
    "bu servis hayatta mı?" diye sorduğu basit kontrol noktası.
    BİLEREK hiçbir ajanı tetiklemiyor - sadece servisin ayakta
    olduğunu (process çalışıyor, FastAPI cevap veriyor) doğrular.
    """
    return {"status": "ok", "service": "living-weather"}


@app.post("/weather", response_model=WeatherContext)
def get_weather(request: WeatherRequest) -> WeatherContext:
    """
    Ana endpoint. Bir konum sorgusunu alır, MasterAgent üzerinden
    5 çekirdek ajanı (Master kendisi dahil) sırayla çalıştırır,
    sonuçta oluşan WeatherContext'i (kara tahta) JSON olarak döner.

    Akış (agent_master.py'deki process_request ile birebir aynı):
      1. Konum -> koordinat çözümü
      2. DI-Agent (gerçek API + devre kesici)
      3. Güven skoru düşükse LLM-Agent (mevsimsel fallback)
      4. AS-Agent (tehlike/uyarı kontrolü, Policy Server üzerinden)
      5. FL-Agent'a kaydet (gelecekte değerlendirilmek üzere)
    """
    context = master_agent.process_request(request.location)

    if context.current_reading is None:
        # Konum tanınmadı (örn. "Wakanda") - agent_master.py'deki
        # SENARYO 2 ile aynı durum. HTTP düzeyinde bunu 404 olarak
        # işaretliyoruz ki istemci (curl/frontend) net bir hata kodu
        # görsün, sessizce boş bir context almasın.
        raise HTTPException(
            status_code=404,
            detail=(
                f"Konum tanınmadı: '{request.location}'. "
                f"Şu an bilinen şehirler: izmir, istanbul, ankara, "
                f"antalya, bodrum."
            ),
        )

    return context


@app.post("/route", response_model=RouteResponse)
def get_route(request: RouteRequest) -> RouteResponse:
    """
    ESNEK AJAN ENDPOINT'İ (Gün 3'te eklendi): RI-Agent üzerinden,
    iki nokta arası hava durumu temelli rota önerisi üretir ve bu
    öneriyi LLM-as-judge ile değerlendirir.

    Akış (agent_master.py'deki process_route_request ile birebir aynı):
      1. origin ve destination için koordinat çözümü
      2. RI-Agent: her iki nokta için DI-Agent çağrısı (aynı paylaşılan
         devre kesici state'i kullanılır)
      3. Rota metni üretimi (LLM dene, başarısız olursa kural tabanlı
         fallback - bkz. agent_ri_agent.py docstring'i)
      4. Üretilen metnin LLM-as-judge ile değerlendirilmesi

    NOT: GEMINI_API_KEY tanımlı değilse veya erişilemezse, sistem
    ÇÖKMEZ - "rule-based-fallback" metne düşer, judge nötr skip eder.
    Bu durumda response'ta text_source="rule-based-fallback" görülür.
    """
    origin_coords, destination_coords, route_response = master_agent.process_route_request(
        request.origin, request.destination
    )

    if route_response is None:
        # origin veya destination tanınmadı - /weather endpoint'indeki
        # 404 mantığıyla AYNI yaklaşım.
        raise HTTPException(
            status_code=404,
            detail=(
                f"Konumlardan biri tanınmadı: origin='{request.origin}', "
                f"destination='{request.destination}'. Şu an bilinen şehirler: "
                f"izmir, istanbul, ankara, antalya, bodrum."
            ),
        )

    payload = route_response.payload
    return RouteResponse(
        origin=payload["origin"],
        destination=payload["destination"],
        route_text=payload["route_text"],
        text_source=payload["text_source"],
        judge_score=payload["judge_score"],
        judge_passed=payload["judge_passed"],
        judge_reason=payload["judge_reason"],
        confidence_score=route_response.confidence_score,
    )


if __name__ == "__main__":
    # Doğrudan "python app/api/main.py" ile de çalıştırılabilir
    # (uvicorn'u programatik başlatır) - ama önerilen yol yukarıdaki
    # docstring'deki "uvicorn app.api.main:app --reload" komutudur,
    # çünkü --reload kod değişikliklerini otomatik yakalar.
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
