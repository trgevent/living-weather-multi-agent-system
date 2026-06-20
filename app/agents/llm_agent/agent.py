"""
app/agents/llm_agent/agent.py
===============================
LLM-Agent (Yerel LLM Modeli Fallback Ajanı)

GÖREV:
  "Zarif Bozunma" felsefesinin İKİNCİ katmanı. DI-Agent'ın hem birincil
  (Open-Meteo) hem yedek kaynağı da güven skoru çok düşükse (örn. < 0.5),
  bu ajan devreye girer. Gerçek bir API'ye bağlanmaz - bunun yerine
  LLM'in kendi "dünya bilgisi"ni kullanarak, o şehir ve mevsim için
  MAKUL bir tahmin üretir.

NEDEN BÖYLE BİR KATMAN GEREKLİ?
  Hayal et: hem Open-Meteo hem VisualCrossing hem NOAA aynı anda çöktü
  (nadir ama mümkün - bölgesel internet kesintisi, DNS sorunu, vb).
  Kullanıcıya "hiçbir veri yok" demek yerine, sistem "İzmir'de Haziran
  ayında genelde 25-30°C civarı olur, açık/parçalı bulutlu olasılığı
  yüksektir" gibi DÜŞÜK GÜVENLE ama YİNE DE FAYDALI bir tahmin verebilir.
  Bu, whitepaper'daki "graceful degradation" (zarif bozunma) ilkesinin
  en uç noktası: sistem asla TAMAMEN sessiz kalmaz, ama düşük güven
  skorunu her zaman AÇIKÇA belirtir (kullanıcıyı yanıltmaz).

GÜVEN PUANI MANTIĞI:
  Bu ajanın ürettiği her tahmin confidence_score=0.2 ile işaretlenir -
  yani "bu gerçek ölçüm değil, tahminsel bir yaklaşım" sinyali sistemin
  her katmanına (AS-Agent, FL-Agent, kullanıcı arayüzü) iletilir.

BİLLİNG/API KEY NOTU:
  Bu prototipte gerçek bir LLM API çağrısı (Gemini vb.) YAPMIYORUZ -
  sebebi basit: bu ajanın devreye girdiği senaryo "her şey çökmüş"
  senaryosu, dış bir LLM API'sine bağımlı olmak bu senaryoda riskli
  olurdu (LLM API'si de çökmüş olabilir). Bunun yerine basit, yerel
  bir mevsimsel-iklim tablosu kullanıyoruz - whitepaper'ın "yerel LLM
  modeli" konseptinin BASİTLEŞTİRİLMİŞ halidir. İstersen ileride
  gerçek bir küçük/yerel modelle (örn. Ollama) değiştirebilirsin.

Kaynak: Living Weather mimarisi (kullanıcı tasarımı)
"""

import time
from datetime import datetime
from typing import Dict, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.models import WeatherReading, DataSourceStatus, AgentResponse


# Basit mevsimsel-iklim referans tablosu (Türkiye şehirleri için örnek).
# Gerçek production'da bu, geçmiş yıllara ait ortalama verilerden
# (örn. açık kaynak iklim veri setlerinden) türetilebilir.
SEASONAL_CLIMATE_TABLE: Dict[str, Dict[int, Tuple[float, str]]] = {
    "izmir": {
        1: (10.0, "rain"), 2: (11.0, "rain"), 3: (14.0, "clear"),
        4: (18.0, "clear"), 5: (23.0, "clear"), 6: (28.0, "clear"),
        7: (31.0, "clear"), 8: (31.0, "clear"), 9: (27.0, "clear"),
        10: (21.0, "clear"), 11: (16.0, "rain"), 12: (12.0, "rain"),
    },
    "istanbul": {
        1: (8.0, "rain"), 2: (8.0, "rain"), 3: (11.0, "rain"),
        4: (15.0, "clear"), 5: (20.0, "clear"), 6: (24.0, "clear"),
        7: (27.0, "clear"), 8: (27.0, "clear"), 9: (23.0, "clear"),
        10: (18.0, "rain"), 11: (13.0, "rain"), 12: (9.0, "rain"),
    },
    "ankara": {
        1: (1.0, "rain"), 2: (3.0, "rain"), 3: (8.0, "clear"),
        4: (13.0, "clear"), 5: (18.0, "clear"), 6: (22.0, "clear"),
        7: (25.0, "clear"), 8: (25.0, "clear"), 9: (20.0, "clear"),
        10: (14.0, "clear"), 11: (7.0, "rain"), 12: (3.0, "rain"),
    },
}

DEFAULT_FALLBACK_TEMP = 18.0  # tablo dışı şehirler için kaba dünya ortalaması


class LLMAgent:
    """
    DI-Agent tamamen başarısız olduğunda devreye giren, mevsimsel
    bilgiye dayalı tahmin üreten son çare ajanı.
    """

    def __init__(self):
        self.agent_name = "LLM-Agent"

    def estimate(self, latitude: float, longitude: float, location_name: str) -> AgentResponse:
        """
        Gerçek ölçüm yerine mevsimsel/coğrafi bilgiye dayalı bir
        tahmin üretir. DÜŞÜK güven skoruyla işaretlenir.
        """
        start = time.monotonic()
        city_key = self._normalize_turkish(location_name)

        month = datetime.utcnow().month
        climate_data = SEASONAL_CLIMATE_TABLE.get(city_key)

        if climate_data and month in climate_data:
            temp, condition = climate_data[month]
            confidence = 0.2  # şehir+ay biliniyor ama hâlâ sadece istatistiksel tahmin
            note = f"'{location_name}' için {month}. ay mevsimsel ortalaması kullanıldı"
        else:
            temp, condition = DEFAULT_FALLBACK_TEMP, "unknown"
            confidence = 0.1  # şehir tanınmıyor, çok daha düşük güven
            note = f"'{location_name}' tanınmıyor, genel dünya ortalaması kullanıldı"

        reading = WeatherReading(
            location=location_name,
            latitude=latitude,
            longitude=longitude,
            temperature_c=temp,
            condition=condition,
            source="llm-agent-seasonal-estimate",
            source_status=DataSourceStatus.CRITICAL,
            confidence_score=confidence,
        )

        elapsed_ms = (time.monotonic() - start) * 1000
        return AgentResponse(
            agent_name=self.agent_name,
            success=True,
            payload={"reading": reading.model_dump(mode="json")},
            confidence_score=confidence,
            error_message=f"DI-Agent veri sağlayamadı, LLM-Agent tahmini devrede. {note}",
            processing_time_ms=elapsed_ms,
        )

    def should_activate(self, di_agent_confidence: float, threshold: float = 0.5) -> bool:
        """
        Master Agent, DI-Agent'ın güven skorunu bu fonksiyona verir.
        Eşiğin altındaysa LLM-Agent devreye girmeli.
        """
        return di_agent_confidence < threshold

    @staticmethod
    def _normalize_turkish(text: str) -> str:
        """
        Türkçe metni, sözlük anahtarı (key) karşılaştırması için
        güvenli şekilde normalize eder.

        NEDEN .lower() KULLANMIYORUZ (doğrudan)?
        Python'un yerleşik .lower() metodu Türkçe'ye özgü bir tuzağa
        sahip: "İ" (büyük noktalı I) karakterini .lower() ile çevirince
        "i" DEĞİL, "i̇" (i + ayrı bir kombine edici nokta karakteri,
        Unicode U+0307) üretir - görünüşte aynı ama 2 karakterden oluşan
        FARKLI bir string. Bu yüzden "İzmir".lower() != "izmir" olur,
        sözlük anahtarı eşleşmesi sessizce başarısız olur (gerçekten
        test ederken bu bug'a yakalandık - "İzmir" tabloda olduğu halde
        "tanınmıyor" sonucu döndü).

        ÇÖZÜM: Açık karakter haritasıyla, .lower() çağırmadan ÖNCE
        Türkçe'ye özgü harfleri manuel olarak ASCII karşılıklarına
        çeviriyoruz, sonra normal .lower() güvenle uygulanabilir.

        Bu sorunu Antigravity'nin code-check skill'i, Day 5 review
        sırasında policy_server.py için zaten işaret etmişti
        (bkz. SKILL.md "Türkçe karakter sorunlarına dikkat" maddesi);
        burada gerçekten karşımıza çıktı, bu da o uyarının ne kadar
        yerinde olduğunu gösteriyor.
        """
        replacements = {
            "İ": "I", "I": "I", "ı": "i",
            "Ğ": "G", "ğ": "g",
            "Ü": "U", "ü": "u",
            "Ş": "S", "ş": "s",
            "Ö": "O", "ö": "o",
            "Ç": "C", "ç": "c",
        }
        result = text.strip()
        for tr_char, ascii_char in replacements.items():
            result = result.replace(tr_char, ascii_char)
        return result.lower()


if __name__ == "__main__":
    agent = LLMAgent()

    print("=== TEST 1: Bilinen şehir (İzmir), mevsimsel tahmin ===")
    result1 = agent.estimate(latitude=38.42, longitude=27.14, location_name="İzmir")
    print(f"Güven skoru: {result1.confidence_score}")
    print(f"Veri: {result1.payload}")
    print(f"Not: {result1.error_message}")

    print("\n=== TEST 2: Tanınmayan şehir, genel fallback ===")
    result2 = agent.estimate(latitude=51.5, longitude=-0.12, location_name="Wakanda")
    print(f"Güven skoru: {result2.confidence_score}")
    print(f"Veri: {result2.payload}")
    print(f"Not: {result2.error_message}")

    print("\n=== TEST 3: should_activate() eşik mantığı ===")
    print(f"DI-Agent confidence=0.3 -> LLM-Agent devreye girmeli mi? {agent.should_activate(0.3)}")
    print(f"DI-Agent confidence=0.8 -> LLM-Agent devreye girmeli mi? {agent.should_activate(0.8)}")
    # Beklenen: 0.3 için True (eşik 0.5'in altında), 0.8 için False
