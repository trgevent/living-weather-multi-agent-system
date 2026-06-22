"""
app/agents/mc_agent/agent.py
==============================
MC-Agent (Micro-Climate Agent) — ESNEK AJAN (zaman kalırsa yazılacak listesinden,
esnek 3'ün ÜÇÜNCÜSÜ ve sonuncusu - RI-Agent ve PA-Agent'tan sonra)

GÖREV:
  İki nokta (örn. bir şehrin iki farklı bölgesi, ya da tamamen farklı
  iki şehir) arasındaki ANLIK mikro-iklim farkını hesaplar - "Westminster
  15°C, Greenwich 13°C" gibi. Yürüyerek/dış mekanda hareket eden bir
  kullanıcının ihtiyacı olan metrikler üzerinden karşılaştırma yapar:
  sıcaklık, yağış (yağmur/kar ayrımıyla), rüzgar, ve bunlardan
  ÇIKARIMLA üretilen bir fırtına riski notu. Bu karşılaştırmayı RI-Agent
  gibi bir LLM'e doğal dilde yorumlatır, sonra LLM-as-judge ile kalite
  kontrolünden geçirir.

NEDEN SERBEST KOORDİNAT (KNOWN_LOCATIONS'a BAĞIMLI DEĞİL)?
  İlk tasarım önerisi MC-Agent'ı Master Agent'ın KNOWN_LOCATIONS
  tablosundaki 5 sabit Türkiye şehriyle sınırlı tutmaktı (örn. "İstanbul:
  Beşiktaş/Kadıköy" gibi sabit ilçe çiftleri). Bu BİLİNÇLİ olarak
  YAPILMADI: Living Weather'ın mikro-iklim yeteneği ileride Atlas
  (kişisel asistan) ve HeyGuide (turist rehberi) projelerinde de
  kullanılması planlanıyor - örn. HeyGuide'ın Benelux turunda bir
  rehber şu an Gent'te (Belçika), 2 saat sonra Lüksemburg'da olabilir.
  Sabit bir Türkiye şehir tablosuna hapsetmek bu gelecekteki kullanım
  amacını YOK EDERDİ.

  Bu yüzden MC-Agent, RI-Agent'ın plan_route(origin, origin_coords,
  destination, destination_coords) imzasıyla AYNI mantıkla, serbest
  (latitude, longitude) koordinat çiftlerini parametre olarak alır.
  KAPSAM SINIRI (bilinçli): bu oturumda Atlas/HeyGuide'a gerçek bir
  entegrasyon YAPILMIYOR, sadece "ileride taşınabilir olacak şekilde"
  tasarlanıyor (RI-Agent'ın Gemini->Anthropic geçişi için provider
  isolation yapmasıyla AYNI prensip - genel tasarım, ama şimdilik
  sadece Living Weather içinde çalışır ve test edilir).

NEDEN BU METRİKLER (sıcaklık, yağış, rüzgar, fırtına riski) - VE
ÇIĞ/SEL NEDEN KAPSAM DIŞI?
  "Yürüyerek hareket eden bir insan" merceğiyle değerlendirildiğinde,
  metrikler GERÇEK VERİ KAYNAĞI olup olmamasına göre ikiye ayrıldı:
    - Open-Meteo'dan doğrudan/çıkarımla elde edilebilenler (KAPSAMDA):
      sıcaklık, yağış (kar/yağmur ayrımı temperature_c<=0 eşiğiyle,
      AS-Agent'ın eşik mantığıyla AYNI desen), rüzgar, ve rüzgar+yağıştan
      çıkarımla üretilen "fırtına riski" notu.
    - Gerçek veri kaynağı OLMAYANLAR (PAS GEÇİLDİ): çığ tehlikesi (kar
      kalınlığı+eğim+zemin verisi gerektirir, Open-Meteo'da yok) ve
      sel/su baskını (hidrolojik veri gerektirir, hava durumu API'sinde
      yok). Bunlar için varsayımsal bir uyarı üretmek, RI-Agent'ın
      LLM-as-judge testinde yakalanan "context hallucination" riskiyle
      (akıcı ama temelsiz metin) AYNI tuzağa düşerdi - BİLİNÇLİ olarak
      MVP kapsamı dışında tutuldu, ileride gerçek bir veri kaynağı
      bulunursa eklenebilir notu olarak saklanıyor.

NEDEN LLM-AS-JUDGE (RI-Agent İLE AYNI PATTERN, PA-Agent'TAN FARKLI)?
  PA-Agent bilinçli olarak kural tabanlı (LLM'siz) bırakıldı - "her ajan
  LLM kullanmak zorunda değil" prensibini göstermek için. MC-Agent için
  TERS bir karar verildi: iki nokta arasındaki farkı doğal dilde,
  bağlama uygun bir şekilde yorumlamak (RI-Agent'ın rota önerisi
  metnindeki "akıcı ama somut tavsiyesiz metin" riskiyle AYNI tuzağı
  barındırır) - bu yüzden RI-Agent'ın İKİ KATMANLI yaklaşımı (LLM dene,
  başarısız olursa kural tabanlı fallback + LLM-as-judge kalite kontrolü)
  burada da AYNEN tekrarlanıyor.

Kaynak: Living Weather mimarisi + HAVA_DURUMU.docx beyin fırtınası
        belgesindeki "Micro-Climate Agent (MC-Agent)" bölümü + Day 5
        whitepaper "Evaluation" (LLM-as-judge) bölümü.
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


# MC-Agent'ın kalite kontrolünde kullandığı rubrik - RI-Agent'ın
# ROUTE_TEXT_RUBRIC'iyle aynı fikir, mikro-iklim bağlamına uyarlandı.
#
# DÜZELTME (21 Haziran 2026, Kaggle Notebook'ta gerçek Gemini'ye karşı
# çalıştırılırken bulundu): İzmir/Bodrum testinde fark gerçekten önemsiz
# (0.6°C) olduğunda, metin bunu doğru tespit edip "özel bir hazırlığa
# gerek yok" dediği halde judge sadece 2.0/5.0 verdi - RI-Agent'taki
# AYNI kök sorun: rubric "fark önemsizse söylesin" diyordu ama bunun
# OLUMLU bir davranış olduğunu açıkça belirtmiyordu, judge bunu zayıf
# bir çıktı gibi okuyabiliyordu. Şimdi açıkça "bu bir eksiklik DEĞİL"
# notu eklendi (RI-Agent'ın rubric düzeltmesiyle AYNI desen).
MICROCLIMATE_TEXT_RUBRIC = (
    "Yorum, iki nokta arasındaki GERÇEK farkı (sıcaklık, yağış, rüzgar) "
    "açıkça belirtiyor mu? Her iki noktanın adı da metinde geçiyor mu? "
    "Fark anlamlıysa (örn. 2°C+ veya yağış/rüzgar durumu değişiyorsa) "
    "yürüyerek hareket eden birine somut bir tavsiye veriyor mu (örn. "
    "'ikinci noktada ek bir kat giyin', 'şemsiye alın')? Fark önemsizse, "
    "bunun doğru bir tespit olduğunu say (bu bir eksiklik DEĞİL) - metnin "
    "bunu açıkça söylemesi ve gereksiz bir alarm/tavsiye ÜRETMEMESİ İYİ "
    "bir davranıştır, düşük puan gerektirmez."
)


class MCAgent:
    """
    İki nokta (serbest koordinat çiftleri) arasındaki mikro-iklim
    farkını hesaplayan ve bu farkı LLM-as-judge ile değerlendirilen
    doğal dilde bir yoruma döken esnek (opsiyonel) ajan.
    """

    def __init__(self, di_agent: Optional[DIAgent] = None, gemini_api_key: Optional[str] = None):
        """
        di_agent: Dependency injection - RI-Agent'taki AYNI desen.
        Master Agent'ta zaten var olan bir DIAgent örneği paylaşılabilir
        (aynı devre kesici state'ini kullanır), verilmezse yeni bir
        DIAgent oluşturulur.
        """
        self.agent_name = "MC-Agent"
        self.di_agent = di_agent or DIAgent()
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

    def compare_locations(
        self,
        point_a_name: str,
        point_a_coords: tuple,
        point_b_name: str,
        point_b_coords: tuple,
    ) -> AgentResponse:
        """
        Ana giriş noktası. point_a_coords/point_b_coords: (latitude,
        longitude) tuple'ları - serbest koordinatlar, herhangi bir sabit
        tabloya bağımlı DEĞİL (bkz. modül docstring'i, "Atlas/HeyGuide"
        gerekçesi).
        """
        start = time.monotonic()

        response_a = self.di_agent.fetch(
            latitude=point_a_coords[0], longitude=point_a_coords[1], location_name=point_a_name
        )
        response_b = self.di_agent.fetch(
            latitude=point_b_coords[0], longitude=point_b_coords[1], location_name=point_b_name
        )

        reading_a = WeatherReading(**response_a.payload["reading"])
        reading_b = WeatherReading(**response_b.payload["reading"])

        diff = self._compute_diff(reading_a, reading_b)

        # KATMAN 1: Mikro-iklim yorumunu üret (LLM dene, başarısız olursa kural tabanlı fallback)
        comparison_text, text_source = self._generate_comparison_text(reading_a, reading_b, diff)

        # KATMAN 2: Üretilen metni LLM-as-judge ile değerlendir
        # (evaluate_with_llm_judge'ın KENDİSİ, GEMINI_API_KEY yoksa zaten
        # nötr "skip" sonucu döndürüyor - RI-Agent'taki AYNI tasarım.)
        #
        # DÜZELTME (21 Haziran 2026): RI-Agent'taki AYNI düzeltme buraya
        # da uygulandı - task_description, judge'a sahte bir zorunluluk
        # ("fark MUTLAKA önemli olmalı" gibi bir izlenim) vermesin diye
        # nötr bir ifadeyle güncellendi.
        judge_result = evaluate_with_llm_judge(
            task_description=(
                f"{point_a_name} ve {point_b_name} arasındaki mikro-iklim "
                f"farkını, yürüyerek hareket eden birine yönelik bir yorumla "
                f"anlat. Fark anlamlıysa somut bir tavsiye ver; fark "
                f"önemsizse bunu açıkça söyle, gereksiz bir uyarı üretme."
            ),
            agent_output=comparison_text,
            rubric=MICROCLIMATE_TEXT_RUBRIC,
            gemini_api_key=self.gemini_api_key,
        )

        elapsed_ms = (time.monotonic() - start) * 1000

        return AgentResponse(
            agent_name=self.agent_name,
            success=True,
            payload={
                "point_a": point_a_name,
                "point_b": point_b_name,
                "reading_a": reading_a.model_dump(mode="json"),
                "reading_b": reading_b.model_dump(mode="json"),
                "temperature_diff_c": diff["temperature_diff_c"],
                "storm_risk": diff["storm_risk"],
                "comparison_text": comparison_text,
                "text_source": text_source,
                "judge_score": judge_result.score,
                "judge_passed": judge_result.passed,
                "judge_reason": judge_result.reason,
            },
            confidence_score=judge_result.score / 5.0,
            processing_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # SAYISAL FARK HESABI (kural tabanlı, LLM gerekmiyor)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_diff(reading_a: WeatherReading, reading_b: WeatherReading) -> dict:
        """
        İki WeatherReading arasındaki sayısal farkı çıkarır. Bu kısım
        TAMAMEN deterministik - whitepaper'ın "doğru aracı doğru işe
        kullan" prensibiyle tutarlı, basit bir çıkarma işlemi için LLM
        gerekmiyor.

        storm_risk: rüzgar VE yağıştan çıkarımla üretilen basit bir
        bayrak - resmi bir severe-weather-alert API'si OLMADAN, AS-Agent'ın
        eşik kontrolü mantığıyla AYNI desen. Çığ/sel burada YOK (bkz.
        modül docstring'i - gerçek veri kaynağı olmadığı için bilinçli
        olarak kapsam dışı).
        """
        temp_diff = round(reading_a.temperature_c - reading_b.temperature_c, 1)

        def is_storm_risk(reading: WeatherReading) -> bool:
            high_wind = bool(reading.wind_speed_kmh and reading.wind_speed_kmh >= 40)
            heavy_rain = bool(reading.precipitation_mm and reading.precipitation_mm > 5)
            return high_wind or heavy_rain or reading.condition == "storm"

        storm_a = is_storm_risk(reading_a)
        storm_b = is_storm_risk(reading_b)

        if storm_a and storm_b:
            storm_risk = "her iki noktada da fırtına riski var"
        elif storm_a:
            storm_risk = f"{reading_a.location}'da fırtına riski var, {reading_b.location}'da yok"
        elif storm_b:
            storm_risk = f"{reading_b.location}'da fırtına riski var, {reading_a.location}'da yok"
        else:
            storm_risk = "iki noktada da fırtına riski görünmüyor"

        return {"temperature_diff_c": temp_diff, "storm_risk": storm_risk}

    # ------------------------------------------------------------------
    # KATMAN 1a: Gerçek LLM çağrısı (sağlayıcıya özel kod SADECE burada)
    # ------------------------------------------------------------------
    def _generate_comparison_text_via_llm(
        self, reading_a: WeatherReading, reading_b: WeatherReading, diff: dict
    ) -> str:
        """
        Gemini'ye iki hava okumasını ve hesaplanmış farkı verip, doğal
        bir mikro-iklim karşılaştırma yorumu yazdırır. SAĞLAYICIYA ÖZEL
        KOD (google.genai importu, client.models.generate_content çağrısı)
        SADECE bu metodun İÇİNDE - RI-Agent'taki AYNI izolasyon prensibi
        (ileride Anthropic'e geçerken sadece bu metodun gövdesi değişir).

        Hata fırlatabilir - çağıran kod (_generate_comparison_text) bunu
        yakalayıp fallback'e düşürüyor, burada try/except YOK (bilerek).
        """
        from google import genai

        client = genai.Client(api_key=self.gemini_api_key)
        prompt = (
            "Bir hava durumu asistanısın. Aşağıda iki nokta için hava durumu "
            "verisi ve aralarındaki hesaplanmış fark var. Bu bilgiye dayanarak, "
            "yürüyerek/dış mekanda hareket eden birine 2-3 cümlelik, doğal bir "
            "mikro-iklim karşılaştırma yorumu yaz.\n\n"
            "ÖNEMLİ KURAL: Fark anlamlıysa (2°C ve üzeri, ya da yağış/rüzgar "
            "durumu farklıysa) SOMUT bir tavsiye ver (örn. 'ikinci noktaya "
            "giderken ek bir kat giyin', 'şemsiyenizi yanınıza alın'). Fark "
            "önemsizse, bunu da açıkça söyle - GEREKSİZ bir uyarı ÜRETME. "
            "Sadece 'hava değişken olabilir' gibi belirsiz bir şey deme, somut "
            "ol. Türkçe yaz.\n\n"
            f"NOKTA A ({reading_a.location}): {reading_a.temperature_c}°C, "
            f"durum: {reading_a.condition}, rüzgar: {reading_a.wind_speed_kmh} km/h, "
            f"yağış: {reading_a.precipitation_mm} mm\n\n"
            f"NOKTA B ({reading_b.location}): {reading_b.temperature_c}°C, "
            f"durum: {reading_b.condition}, rüzgar: {reading_b.wind_speed_kmh} km/h, "
            f"yağış: {reading_b.precipitation_mm} mm\n\n"
            f"HESAPLANMIŞ FARK: sıcaklık farkı {diff['temperature_diff_c']}°C "
            f"(A eksi B), {diff['storm_risk']}."
        )

        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

        # NOT (21 Haziran 2026 eklendi): RI-Agent'taki AYNI gerekçe -
        # Gemini ücretsiz katmanının dakikalık (15 RPM) sınırına çarpma
        # riskini düşürmek için kısa bir bekleme ekleniyor (bkz.
        # CHAPTER_NOTES.md, "Gemini ücretsiz katman GÜNLÜK 20 istek
        # kotası" notu - bu, GÜNLÜK kotayı çözmez, sadece dakikalık
        # sınıra çarpma riskini düşürür).
        time.sleep(4.5)

        text = (response.text or "").strip()
        if not text:
            raise ValueError("Gemini boş bir cevap döndürdü")
        return text

    # ------------------------------------------------------------------
    # KATMAN 1b: Kural tabanlı fallback (API key yok / network yok / hata)
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_comparison_text_rule_based(
        reading_a: WeatherReading, reading_b: WeatherReading, diff: dict
    ) -> str:
        """
        if/else şablon tabanlı, %100 deterministik karşılaştırma metni.
        LLM çağrısı tamamen BAŞARISIZ olursa devreye girer - sistem
        ÇÖKMEZ, RI-Agent'ın kural tabanlı fallback'iyle AYNI felsefe.
        """
        abs_diff = abs(diff["temperature_diff_c"])

        if abs_diff < 1.0:
            magnitude_note = "sıcaklık açısından gözle görülür bir fark yok"
        elif abs_diff < 3.0:
            magnitude_note = "hafif bir sıcaklık farkı var, ek bir kat gerekmeyebilir"
        else:
            magnitude_note = "belirgin bir sıcaklık farkı var, ek bir kat giymeniz önerilir"

        precip_note = ""
        if reading_a.precipitation_mm and reading_a.precipitation_mm > 0 and not (
            reading_b.precipitation_mm and reading_b.precipitation_mm > 0
        ):
            precip_note = f" {reading_a.location}'da yağış var, {reading_b.location}'da yok - şemsiye bulundurun."
        elif reading_b.precipitation_mm and reading_b.precipitation_mm > 0 and not (
            reading_a.precipitation_mm and reading_a.precipitation_mm > 0
        ):
            precip_note = f" {reading_b.location}'da yağış var, {reading_a.location}'da yok - şemsiye bulundurun."

        # DÜZELTME (MC-Agent ilk test sırasında bulundu, 21 Haziran 2026):
        # Python'ın .capitalize() metodu önce TÜM string'i küçük harfe
        # çevirip sonra ilk harfi büyütüyor - "İzmir" geçen bir metinde
        # bu, CHAPTER_NOTES.md'deki "Türkçe İ .lower() tuzağı" bug'ının
        # AYNISINI üretiyordu ("İzmir" -> "i̇zmir", bozuk kombine karakter).
        # Bunun yerine SADECE ilk karakteri büyütüyoruz, string'in kalanına
        # HİÇ dokunmuyoruz - storm_risk metninin içindeki "İzmir"/"Ankara"
        # gibi özel isimler bozulmadan kalır.
        storm_risk_text = diff["storm_risk"]
        storm_risk_capitalized = storm_risk_text[0].upper() + storm_risk_text[1:] if storm_risk_text else storm_risk_text

        return (
            f"{reading_a.location}: {reading_a.temperature_c}°C, {reading_a.condition}. "
            f"{reading_b.location}: {reading_b.temperature_c}°C, {reading_b.condition}. "
            f"Fark: {diff['temperature_diff_c']}°C ({magnitude_note}). "
            f"{storm_risk_capitalized}.{precip_note}"
        )

    # ------------------------------------------------------------------
    # KATMAN 1 (birleştirici): LLM dene, başarısız olursa fallback'e düş
    # ------------------------------------------------------------------
    def _generate_comparison_text(
        self, reading_a: WeatherReading, reading_b: WeatherReading, diff: dict
    ) -> tuple[str, str]:
        """
        Döner: (metin, kaynak) - kaynak "llm-generated" veya
        "rule-based-fallback" olur. RI-Agent'taki AYNI desen.
        """
        if not self.gemini_api_key:
            return (
                self._generate_comparison_text_rule_based(reading_a, reading_b, diff),
                "rule-based-fallback",
            )

        try:
            text = self._generate_comparison_text_via_llm(reading_a, reading_b, diff)
            return text, "llm-generated"
        except Exception:
            # ImportError, API hatası, timeout, boş cevap - HEPSİ için
            # aynı davranış: fallback'e düş. MC-Agent'ın kullanıcıya
            # döndüğü sonuç ÇÖKMEMELİ.
            return (
                self._generate_comparison_text_rule_based(reading_a, reading_b, diff),
                "rule-based-fallback",
            )


if __name__ == "__main__":
    print("=== TEST 1: MC-Agent (İzmir vs Bodrum mikro-iklim karşılaştırması) ===")
    print("NOT: GEMINI_API_KEY tanımlıysa gerçek LLM yorumu üretilir ve gerçek")
    print("LLM-as-judge değerlendirmesi yapılır. Tanımlı değilse, kural tabanlı")
    print("fallback metne düşülür ve judge 'skip' eder.\n")

    agent = MCAgent()
    print(f"GEMINI_API_KEY tanımlı mı: {bool(agent.gemini_api_key)}\n")

    result = agent.compare_locations(
        point_a_name="İzmir", point_a_coords=(38.42, 27.14),
        point_b_name="Bodrum", point_b_coords=(37.04, 27.43),
    )

    print(f"Metin kaynağı: {result.payload['text_source']}")
    print(f"Karşılaştırma: {result.payload['comparison_text']}\n")
    print(f"Sıcaklık farkı (A-B): {result.payload['temperature_diff_c']}°C")
    print(f"Fırtına riski notu: {result.payload['storm_risk']}")
    print(f"Judge skoru: {result.payload['judge_score']} / 5.0")
    print(f"Judge geçti mi: {result.payload['judge_passed']}")
    print(f"Judge gerekçesi: {result.payload['judge_reason']}")
    print(f"\nİşlem süresi: {result.processing_time_ms:.1f} ms")
    print(f"Genel güven skoru: {result.confidence_score:.2f}")

    print("\n=== TEST 2: Serbest koordinat - Türkiye dışı bir çift (Gent vs Lüksemburg) ===")
    print("NOT: Bu test, MC-Agent'ın KNOWN_LOCATIONS tablosuna BAĞIMLI OLMADIĞINI,")
    print("herhangi bir koordinat çiftiyle çalıştığını kanıtlıyor (Atlas/HeyGuide")
    print("senaryosu - bkz. modül docstring'i).\n")

    result2 = agent.compare_locations(
        point_a_name="Gent", point_a_coords=(51.05, 3.72),
        point_b_name="Lüksemburg", point_b_coords=(49.61, 6.13),
    )
    print(f"Metin kaynağı: {result2.payload['text_source']}")
    print(f"Karşılaştırma: {result2.payload['comparison_text']}")
    print(f"Sıcaklık farkı (A-B): {result2.payload['temperature_diff_c']}°C")
    print(">>> Dikkat: Gent/Lüksemburg, Master Agent'ın KNOWN_LOCATIONS tablosunda")
    print(">>> HİÇ YOK - MC-Agent yine de çalıştı, çünkü serbest koordinat alıyor.")

    print("\n=== TEST 3: API key'i bilerek geçersiz yaparak fallback'i zorlama ===")
    broken_agent = MCAgent(gemini_api_key="gecersiz-anahtar-12345")
    result3 = broken_agent.compare_locations(
        point_a_name="İzmir", point_a_coords=(38.42, 27.14),
        point_b_name="Ankara", point_b_coords=(39.93, 32.86),
    )
    print(f"Metin kaynağı: {result3.payload['text_source']}")
    print(f"Karşılaştırma: {result3.payload['comparison_text']}")
    print(">>> Beklenen: text_source='rule-based-fallback' (geçersiz key ile")
    print(">>> Gemini çağrısı hata fırlatır, sistem ÇÖKMEZ, kural tabanlı metne düşer).")
