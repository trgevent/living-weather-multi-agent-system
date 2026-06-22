"""
app/agents/ri_agent/agent.py
==============================
RI-Agent (Route Intelligence Agent) — ESNEK AJAN (zaman kalırsa yazılacak listesinden)

GÖREV:
  Kullanıcı "A'dan B'ye giderken hava nasıl olacak?" diye sorduğunda,
  başlangıç (origin) ve bitiş (destination) noktaları için hava
  durumunu çeker, bir ROTA ÖNERİSİ METNİ üretir, ve bu metni Day 5
  whitepaper'ın "LLM-as-judge" pattern'i ile KALİTE KONTROLÜNDEN
  geçirir.

NEDEN BU AJAN ÖNEMLİ (mimari kanıt açısından)?
  evaluation_engine.py'de ZATEN yazılmış ama hiçbir ajan tarafından
  GERÇEKTEN KULLANILMAMIŞ bir fonksiyon vardı: evaluate_with_llm_judge.
  FL-Agent sadece evaluate_numeric_tolerance (tolerans bandı) kullanıyor
  - yani whitepaper'ın 3 evaluation pattern'inden (binary test, tolerans
  bandı, LLM-as-judge) sadece 2'si kanıtlanmıştı. RI-Agent, ÜÇÜNCÜSÜNÜ
  (LLM-as-judge) ilk kez gerçek bir senaryoda devreye sokuyor.

  whitepaper'ın kendi örneği zaten TAM OLARAK bu senaryoydu (bkz.
  evaluation_engine.py docstring'i): "RI-Agent bir rota önerisi metni
  üretti, bunun 'doğru' olup olmadığını regex ile ölçemezsin - bir
  LLM'e 'bu öneri mantıklı mı, güvenli mi, eksik bir uyarı var mı?'
  diye sorman gerekir."

NEDEN İKİ KATMANLI FALLBACK (LLM yoksa ne olur)?
  Bu ajan İKİ farklı noktada dış bir LLM API'sine (Gemini) bağımlı
  olabilir: (1) rota metnini ÜRETİRKEN, (2) o metni DEĞERLENDİRİRKEN.
  Kullanıcının konumu/ağı API'ye erişemiyorsa (internet yok, API key
  tanımlı değil, kota aşıldı), sistem ÇÖKMEMELİ - bu, projenin DI-Agent
  ve LLM-Agent'tan beri sürdürdüğü "Zarif Bozunma" felsefesinin
  ÜÇÜNCÜ bir noktada tekrar kanıtı:

    KATMAN 1 (metin üretimi):
      Gemini çağrısı başarılı  -> text_source="llm-generated"
      Gemini çağrısı başarısız -> text_source="rule-based-fallback"
                                   (kural tabanlı, if/else şablon metin)

    KATMAN 2 (metin değerlendirmesi, evaluate_with_llm_judge içinde
    ZATEN var olan davranış):
      GEMINI_API_KEY yok/google-genai kurulu değil -> nötr "skip" skoru
      (sistem durmaz, sadece değerlendirme atlanır)

  Bu sayede RI-Agent, internet/API erişimi OLMAYAN bir makinede de
  ÇALIŞIR (düşük güvenle, kural tabanlı metinle) - tam DI-Agent'ın
  Open-Meteo çökünce LLM-Agent'a düşmesiyle AYNI mimari desen.

NEDEN GEMINI (ANTHROPIC DEĞİL) - VE İLERİDE DEĞİŞTİRMEK NE KADAR
KOLAY OLACAK?
  Şu an proje genelinde (policy_server.py, evaluation_engine.py)
  zaten Gemini API kullanılıyor (GEMINI_API_KEY, Day 1'de test edildi).
  Tutarlılık için RI-Agent de Gemini kullanıyor. AMA: kullanıcının
  Atlas (aile asistanı) projesinde Anthropic API kullanılıyor ve
  ileride Living Weather'ı Anthropic'e geçirme isteği var.

  Bu geçişi KOLAYLAŞTIRMAK için, "LLM'e rota metni yazdırma" işlemi
  bu dosyada AYRI BİR FONKSİYONA (_generate_route_text_via_llm)
  izole edildi - RI-Agent'ın ana akışı (plan_route metodu) bu
  fonksiyonu SAĞLAYICIDAN BAĞIMSIZ bir imza ile çağırıyor (origin
  reading + destination reading al, metin döndür). İleride Anthropic'e
  geçerken SADECE bu fonksiyonun İÇİ değişir (örn. genai.Client yerine
  anthropic.Anthropic() çağrısı) - plan_route metodu, AgentResponse
  yapısı, çağıran kod (agent_master.py) HİÇ DEĞİŞMEZ. Bu, Redis
  entegrasyonunda RedisCircuitBreaker'ın eski CircuitBreaker ile aynı
  public arayüzü koruması ile AYNI tasarım prensibidir.

Kaynak: Living Weather mimarisi + Day 5 whitepaper "Evaluation"
        bölümünün LLM-as-judge örneği (s.29-30, evaluation_engine.py
        docstring'inde zaten alıntılanmış senaryo).
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.models import WeatherReading, AgentResponse
from app.core.evaluation_engine import evaluate_with_llm_judge
from app.agents.di_agent.agent import DIAgent


# RI-Agent'ın kalite kontrolünde kullandığı rubrik - whitepaper'ın
# evaluation_engine.py'deki ÖRNEK rubrikle aynı fikir, biraz genişletildi.
#
# DÜZELTME (21 Haziran 2026, task_description düzeltmesiyle birlikte):
# Son cümleye "hava açıksa bunu doğru bir tespit olarak say" notu
# eklendi - judge'ın "kötü hava koşulu YOK" durumunu bir EKSİKLİK gibi
# değerlendirmesini önlemek için (bkz. agent.py'deki task_description
# düzeltmesinin gerekçesi).
ROUTE_TEXT_RUBRIC = (
    "Öneri, varsa kötü hava koşullarını (yağmur, fırtına, aşırı sıcak/soğuk, "
    "şiddetli rüzgar) açıkça belirtiyor mu? Hava açık/normalse, bunun doğru "
    "bir tespit olduğunu say (bu bir eksiklik DEĞİL). Başlangıç VE bitiş "
    "noktasının her ikisi de metinde geçiyor mu? Net ve eyleme geçirilebilir "
    "mi (örn. 'yağmurluk al', 'güneş gözlüğü al' gibi somut bir tavsiye var mı)?"
)


class RIAgent:
    """
    İki nokta (origin, destination) arasında hava durumu temelli bir
    rota önerisi üreten ve bu öneriyi LLM-as-judge ile değerlendiren
    esnek (opsiyonel) ajan.
    """

    def __init__(self, di_agent: Optional[DIAgent] = None, gemini_api_key: Optional[str] = None):
        """
        di_agent: Dependency injection - testlerde/Master Agent'ta
        zaten var olan bir DIAgent örneği paylaşılabilir (aynı devre
        kesici state'ini kullanır), verilmezse yeni bir DIAgent oluşturulur.
        """
        self.agent_name = "RI-Agent"
        self.di_agent = di_agent or DIAgent()
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

    def plan_route(self, origin: str, origin_coords: tuple, destination: str, destination_coords: tuple) -> AgentResponse:
        """
        Ana giriş noktası. origin_coords/destination_coords: (latitude, longitude)
        tuple'ları - Master Agent'taki KNOWN_LOCATIONS tablosundan gelir.
        """
        start = time.monotonic()

        origin_response = self.di_agent.fetch(
            latitude=origin_coords[0], longitude=origin_coords[1], location_name=origin
        )
        destination_response = self.di_agent.fetch(
            latitude=destination_coords[0], longitude=destination_coords[1], location_name=destination
        )

        origin_reading = WeatherReading(**origin_response.payload["reading"])
        destination_reading = WeatherReading(**destination_response.payload["reading"])

        # KATMAN 1: Rota metnini üret (LLM dene, başarısız olursa kural tabanlı fallback)
        route_text, text_source = self._generate_route_text(origin_reading, destination_reading)

        # KATMAN 2: Üretilen metni LLM-as-judge ile değerlendir
        # (evaluate_with_llm_judge'ın KENDİSİ, GEMINI_API_KEY yoksa
        # zaten nötr "skip" sonucu döndürüyor - burada ek bir try/except
        # gerekmiyor, fonksiyon zaten FAIL-SAFE tasarlanmış.)
        #
        # DÜZELTME (Notebook'ta Kaggle Secrets ile gerçek Gemini'ye karşı
        # çalıştırılırken bulundu, 21 Haziran 2026): task_description
        # eskiden "...hava durumu UYARISI İÇEREN bir rota önerisi yaz"
        # diyordu - bu, judge'a "görev tanımı UYARI ZORUNLU" sinyali
        # veriyordu. Hava GERÇEKTEN açık/temiz olduğunda (uyarı vermeye
        # gerek olmayan, DOĞRU bir durumda), judge metni "görev tanımına
        # uymadı" diye 1.0/5.0 ile reddetti - HALBUKİ metin zaten somut
        # tavsiyeler içeriyordu (güneş gözlüğü, su). Sorun rubric'te değil
        # (rubric zaten "VARSA kötü hava koşullarını belirt" diyordu),
        # task_description'da kendi içinde çelişkiliydi. task_description
        # artık rubric'le AYNI nötr ifadeyi kullanıyor - "varsa uyarı ver,
        # yoksa somut tavsiye ver" - judge'a sahte bir zorunluluk dayatmıyor.
        judge_result = evaluate_with_llm_judge(
            task_description=(
                f"{origin}'dan {destination}'a giderken hava durumuna dayalı "
                f"bir rota önerisi yaz. Kötü hava koşulları varsa bunları "
                f"açıkça belirt; hava açık/normalse bunu söyle ve yine de "
                f"somut bir tavsiye ver."
            ),
            agent_output=route_text,
            rubric=ROUTE_TEXT_RUBRIC,
            gemini_api_key=self.gemini_api_key,
        )

        elapsed_ms = (time.monotonic() - start) * 1000

        return AgentResponse(
            agent_name=self.agent_name,
            success=True,
            payload={
                "origin": origin,
                "destination": destination,
                "route_text": route_text,
                "text_source": text_source,
                "origin_reading": origin_reading.model_dump(mode="json"),
                "destination_reading": destination_reading.model_dump(mode="json"),
                "judge_score": judge_result.score,
                "judge_passed": judge_result.passed,
                "judge_reason": judge_result.reason,
            },
            confidence_score=judge_result.score / 5.0,
            processing_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # KATMAN 1a: Gerçek LLM çağrısı (sağlayıcıya özel kod SADECE burada)
    # ------------------------------------------------------------------
    def _generate_route_text_via_llm(self, origin_reading: WeatherReading, destination_reading: WeatherReading) -> str:
        """
        Gemini'ye iki hava okumasını verip, doğal bir rota önerisi
        metni yazdırır. SAĞLAYICIYA ÖZEL KOD (google.genai importu,
        client.models.generate_content çağrısı) SADECE bu metodun
        İÇİNDE - ileride Anthropic'e geçerken SADECE bu metodun
        gövdesi değişir, plan_route() ve diğer her şey aynı kalır.

        Hata fırlatabilir (ImportError, API hatası, timeout vb.) -
        çağıran kod (_generate_route_text) bunu yakalayıp fallback'e
        düşürüyor, burada try/except YOK (bilerek - sorumluluk ayrımı).
        """
        from google import genai

        client = genai.Client(api_key=self.gemini_api_key)
        prompt = (
            "Bir hava durumu asistanısın. Aşağıdaki iki nokta için hava "
            "durumu verisi var. Bu bilgiye dayanarak, yolculuk yapacak "
            "birine 2-3 cümlelik, doğal bir rota önerisi yaz. Varsa kötü "
            "hava koşullarını (yağmur, fırtına, aşırı sıcak/soğuk, şiddetli "
            "rüzgar) AÇIKÇA belirt ve somut bir tavsiye ver (örn. yağmurluk, "
            "erken çıkış vb.).\n\n"
            "ÖNEMLİ KURAL: Hava durumu tamamen açık/temiz (clear) olsa bile "
            "SADECE 'hava güzel, iyi yolculuklar' deme - bu yetersiz sayılır. "
            "Her durumda yolcuya SOMUT ve EYLEME GEÇİRİLEBİLİR en az bir "
            "tavsiye ver. Örnek: hava açıksa 'güneş gözlüğü ve su "
            "bulundurun', sıcaksa 'klimayı kontrol edin, bol su için', "
            "rüzgarlıysa 'direksiyonu sıkı tutun' gibi somut, eyleme dönük "
            "bir öneri MUTLAKA ekle. Türkçe yaz.\n\n"
            f"BAŞLANGIÇ ({origin_reading.location}): {origin_reading.temperature_c}°C, "
            f"durum: {origin_reading.condition}, rüzgar: {origin_reading.wind_speed_kmh} km/h, "
            f"yağış: {origin_reading.precipitation_mm} mm\n\n"
            f"BİTİŞ ({destination_reading.location}): {destination_reading.temperature_c}°C, "
            f"durum: {destination_reading.condition}, rüzgar: {destination_reading.wind_speed_kmh} km/h, "
            f"yağış: {destination_reading.precipitation_mm} mm"
        )

        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

        # NOT (21 Haziran 2026 eklendi): Gemini ücretsiz katmanının
        # DAKİKALIK (15 RPM) ve GÜNLÜK (20/gün) kota sınırları var.
        # Kaggle Notebook'ta RI-Agent + MC-Agent toplam 4 gerçek Gemini
        # çağrısı yapıyor (her ajan 2 kez: normal + bozuk-key testi).
        # "Save & Run All" bu çağrıları milisaniyeler içinde art arda
        # tetikleyebiliyor - dakikalık sınıra çarpma riskini azaltmak
        # için her çağrıdan sonra kısa bir bekleme ekleniyor. Bu, GÜNLÜK
        # kota sorununu ÇÖZMEZ (o ayrı bir kısıt, bkz. CHAPTER_NOTES.md)
        # ama DAKİKALIK sınıra çarpma riskini düşürür.
        time.sleep(4.5)

        text = (response.text or "").strip()
        if not text:
            raise ValueError("Gemini boş bir cevap döndürdü")
        return text

    # ------------------------------------------------------------------
    # KATMAN 1b: Kural tabanlı fallback (API key yok / network yok / hata)
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_route_text_rule_based(origin_reading: WeatherReading, destination_reading: WeatherReading) -> str:
        """
        if/else şablon tabanlı, %100 deterministik rota metni.
        LLM çağrısı tamamen BAŞARISIZ olursa devreye girer - sistem
        ÇÖKMEZ, sadece daha "mekanik" bir metin üretir. Bu, DI-Agent'ın
        Open-Meteo çökünce "fallback-degraded" okumasına düşmesiyle
        AYNI felsefe: düşük kalite ama YİNE DE FAYDALI bir çıktı.
        """
        def describe(reading: WeatherReading) -> str:
            warnings = []
            if reading.precipitation_mm and reading.precipitation_mm > 0:
                warnings.append("yağış bekleniyor, yağmurluk/şemsiye almanız önerilir")
            if reading.wind_speed_kmh and reading.wind_speed_kmh >= 40:
                warnings.append("şiddetli rüzgar bekleniyor")
            if reading.temperature_c >= 35:
                warnings.append("aşırı sıcak bekleniyor, bol su tüketin")
            if reading.temperature_c <= 0:
                warnings.append("don/aşırı soğuk bekleniyor")
            if not warnings:
                # DÜZELTME (RI-Agent prompt iyileştirmesi, 20 Haziran 2026):
                # Hava açık/temizken de SOMUT bir tavsiye ekleniyor - LLM-as
                # -judge testinde "sadece 'hava güzel' demek yetersiz" diye
                # işaretlenmişti (gerçek test sırasında bulundu). Kural
                # tabanlı fallback de aynı standarda uymalı, tutarlılık için.
                warnings.append("güneş gözlüğü ve su bulundurmanız önerilir")
            warning_text = f" ({', '.join(warnings)})" if warnings else ""
            return f"{reading.location}: {reading.temperature_c}°C, {reading.condition}{warning_text}"

        return (
            f"Rota bilgisi - {describe(origin_reading)}. "
            f"Varış noktası {describe(destination_reading)}. "
            f"Yolculuğunuzu bu bilgilere göre planlayabilirsiniz."
        )

    # ------------------------------------------------------------------
    # KATMAN 1 (birleştirici): LLM dene, başarısız olursa fallback'e düş
    # ------------------------------------------------------------------
    def _generate_route_text(self, origin_reading: WeatherReading, destination_reading: WeatherReading) -> tuple[str, str]:
        """
        Döner: (metin, kaynak) - kaynak "llm-generated" veya
        "rule-based-fallback" olur. Master Agent / kullanıcı arayüzü,
        bu kaynağı görerek metnin ne kadar güvenilir olduğunu anlayabilir
        (tam confidence_score mantığının DI-Agent/LLM-Agent'ta yaptığı gibi).
        """
        if not self.gemini_api_key:
            return (
                self._generate_route_text_rule_based(origin_reading, destination_reading),
                "rule-based-fallback",
            )

        try:
            text = self._generate_route_text_via_llm(origin_reading, destination_reading)
            return text, "llm-generated"
        except Exception:
            # ImportError (google-genai kurulu değil), API hatası, timeout,
            # boş cevap (ValueError) - HEPSİ için aynı davranış: fallback'e düş.
            # Hangi hata olduğunu loglamak ileride faydalı olabilir, ama
            # RI-Agent'ın kullanıcıya döndüğü sonuç ÇÖKMEMELİ.
            return (
                self._generate_route_text_rule_based(origin_reading, destination_reading),
                "rule-based-fallback",
            )


if __name__ == "__main__":
    print("=== TEST: RI-Agent (İzmir -> Bodrum rota önerisi) ===")
    print("NOT: GEMINI_API_KEY tanımlıysa gerçek LLM metni üretilir ve")
    print("gerçek LLM-as-judge değerlendirmesi yapılır. Tanımlı değilse,")
    print("kural tabanlı fallback metne düşülür ve judge 'skip' eder.\n")

    agent = RIAgent()

    print(f"GEMINI_API_KEY tanımlı mı: {bool(agent.gemini_api_key)}\n")

    result = agent.plan_route(
        origin="İzmir", origin_coords=(38.42, 27.14),
        destination="Bodrum", destination_coords=(37.04, 27.43),
    )

    print(f"Metin kaynağı: {result.payload['text_source']}")
    print(f"Rota metni: {result.payload['route_text']}\n")
    print(f"Judge skoru: {result.payload['judge_score']} / 5.0")
    print(f"Judge geçti mi: {result.payload['judge_passed']}")
    print(f"Judge gerekçesi: {result.payload['judge_reason']}")
    print(f"\nİşlem süresi: {result.processing_time_ms:.1f} ms")
    print(f"Genel güven skoru: {result.confidence_score:.2f}")

    print("\n=== TEST 2: API key'i bilerek geçersiz yaparak fallback'i zorlama ===")
    broken_agent = RIAgent(gemini_api_key="gecersiz-anahtar-12345")
    result2 = broken_agent.plan_route(
        origin="İzmir", origin_coords=(38.42, 27.14),
        destination="Ankara", destination_coords=(39.93, 32.86),
    )
    print(f"Metin kaynağı: {result2.payload['text_source']}")
    print(f"Rota metni: {result2.payload['route_text']}")
    print(">>> Beklenen: text_source='rule-based-fallback' (geçersiz key ile")
    print(">>> Gemini çağrısı hata fırlatır, sistem ÇÖKMEZ, kural tabanlı metne düşer).")
