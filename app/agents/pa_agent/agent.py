"""
app/agents/pa_agent/agent.py
==============================
PA-Agent (Personal Advisory Agent) — ESNEK AJAN

GÖREV:
  DI-Agent'ın hava verisini alır, kullanıcıya KİŞİSELLEŞTİRİLMİŞ
  öneriler üretir: kıyafet, sağlık, aktivite. Örnek: "12°C, yağmur
  %80 → su geçirmez mont, katmanlı giyin" veya "UV indeksi yüksek,
  güneş kremi ve gözlük öner".

NEDEN BU AJAN ÖNEMLİ (mimari kanıt açısından)?
  core/context_resolver.py ZATEN Gün 1'den beri yazılıydı ama hiçbir
  ajan tarafından GERÇEKTEN KULLANILMAMIŞTI - whitepaper'ın "Context
  Hygiene / Prompt Sanitization" pattern'i sadece kodda duruyordu,
  hiçbir gerçek senaryoda kanıtlanmamıştı. PA-Agent, bunu ilk kez
  devreye sokuyor: kullanıcıya gösterilecek öneri metni bir ŞABLON
  olarak tutuluyor ([[LOCATION]], [[TEMP_C]] gibi placeholder'larla),
  gerçek değerler sadece SON ANDA (resolve_context ile) enjekte
  ediliyor.

  Bu, RI-Agent'ın evaluate_with_llm_judge'ı ilk kullanması ile AYNI
  hikaye: yazılmış ama kullanılmamış bir whitepaper pattern'inin,
  doğal bir senaryoyla gerçek bir ajanda kanıtlanması.

NEDEN KURAL TABANLI (LLM ÇAĞRISI YOK)?
  RI-Agent zaten whitepaper'ın "LLM-as-judge" pattern'ini kanıtladı.
  PA-Agent için aynı şeyi tekrarlamak yerine, BİLİNÇLİ olarak basit
  if/else kurallarına dayalı bir yaklaşım seçildi - bu hem hızlı/
  ücretsiz (dış API çağrısı yok) hem de "her ajan bir LLM çağırmak
  zorunda değil" prensibini gösteriyor. Kıyafet/sağlık önerisi gibi
  deterministik eşik tabanlı bir karar (örn. "sıcaklık < 5°C ise
  kalın mont öner") zaten LLM gerektirmeyen bir problem - whitepaper'ın
  "doğru aracı doğru işe kullan" felsefesiyle tutarlı.

KAYNAK BELGE:
  Kullanıcının "HAVA_DURUMU.docx" beyin fırtınası dosyasındaki
  "Personal Advisory Agent (PA-Agent)" bölümünden uyarlanmıştır
  (kıyafet önerisi, sağlık önerisi, aktivite önerisi üç alt-kategori
  olarak orada da tanımlıydı). MVP kapsamı, aynı belgenin "Bu Projeyi
  Bitirmiş Saymak İçin Minimum Gereksinimler" listesindeki "PA-Agent
  kıyafet önerisi veriyor" maddesiyle uyumlu tutuldu - havacılık/
  denizcilik gibi daha büyük kapsamlı uzantılar (RI-Agent'a eklenmesi
  düşünülen ama zaman/risk nedeniyle ERTELENEN METAR/crosswind
  hesaplamaları) bu MVP'ye dahil EDİLMEDİ.

Kaynak: Living Weather mimarisi + Day 5 whitepaper "Implementing a
        Dynamic ContextResolver" bölümü (s.33-34).
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.models import WeatherReading, AgentResponse
from app.core.context_resolver import resolve_context


class PAAgent:
    """
    Hava verisine dayalı kişisel öneri üreten ajan. Kıyafet, sağlık
    ve aktivite olmak üzere 3 ayrı kategori önerisi üretir, her biri
    [[PLACEHOLDER]] şablonları üzerinden context_resolver.py ile
    çözülür.
    """

    def __init__(self):
        self.agent_name = "PA-Agent"

    def advise(self, reading: WeatherReading, user_name: Optional[str] = None) -> AgentResponse:
        """
        Ana giriş noktası. reading: DI-Agent veya LLM-Agent'tan gelen
        bir WeatherReading. user_name: opsiyonel, verilirse önerilerde
        kişiselleştirme için kullanılır (context_resolver üzerinden).

        NOT: reading.confidence_score düşükse (örn. LLM-Agent'tan
        gelen mevsimsel tahmin), bu BİLGİ kullanıcıya İLETİLİR (DI-Agent/
        AS-Agent'taki gibi şeffaflık ilkesi) ama öneri ÜRETİLMEYE
        DEVAM EDİLİR - AS-Agent'taki "düşük güvenle otomatik alert
        gönderme" kısıtlaması burada GEÇERLİ DEĞİL, çünkü PA-Agent'ın
        önerileri bir GÜVENLİK aksiyonu değil, bir KONFOR/rahatlık
        önerisi - yanlış olsa bile en kötü ihtimalle kullanıcı fazla
        kalın giyinir, bu AS-Agent'ın yanlış alarm riskiyle KIYASLANAMAZ
        bir risk seviyesi. (Bu ayrım CHAPTER_NOTES.md'ye not edilmeli:
        "her düşük güven durumu otomatik olarak aksiyon engellemeli"
        DEĞİL, asıl soru "bu aksiyon yanlış olursa ne kadar zarar
        verir" - AS-Agent'ın alert'i için zarar yüksek, PA-Agent'ın
        kıyafet önerisi için zarar düşük.)
        """
        clothing_advice = self._build_clothing_advice(reading, user_name)
        health_advice = self._build_health_advice(reading, user_name)
        activity_advice = self._build_activity_advice(reading, user_name)

        return AgentResponse(
            agent_name=self.agent_name,
            success=True,
            payload={
                "clothing_advice": clothing_advice,
                "health_advice": health_advice,
                "activity_advice": activity_advice,
                "based_on_source": reading.source,
                "based_on_confidence": reading.confidence_score,
            },
            confidence_score=reading.confidence_score,
            error_message=(
                None if reading.confidence_score >= 0.5 else
                f"Not: Bu öneriler düşük güvenilirlikli bir tahmine "
                f"({reading.source}, güven: {reading.confidence_score}) "
                f"dayanıyor, gerçek ölçüm değil."
            ),
        )

    # ------------------------------------------------------------------
    # KIYAFET ÖNERİSİ
    # ------------------------------------------------------------------
    def _build_clothing_advice(self, reading: WeatherReading, user_name: Optional[str]) -> str:
        """
        Sıcaklık + yağış + rüzgara göre kural tabanlı kıyafet önerisi.
        Şablon, [[LOCATION]] ve (varsa) [[USER_NAME]] placeholder'ları
        ile yazılıyor, sonra resolve_context ile çözülüyor - bu, PA-Agent'ın
        context_resolver.py'yi GERÇEKTEN kullandığı nokta.
        """
        items = []

        temp = reading.temperature_c
        if temp <= 0:
            items.append("kalın mont, atkı, bere ve eldiven")
        elif temp <= 10:
            items.append("kalın mont veya kaban, katmanlı giyin")
        elif temp <= 18:
            items.append("hafif ceket veya hırka")
        elif temp <= 27:
            items.append("hafif, nefes alan kıyafetler")
        else:
            items.append("çok hafif, terletmeyen kumaşlar")

        if reading.precipitation_mm and reading.precipitation_mm > 0:
            items.append("su geçirmez mont veya şemsiye")

        if reading.wind_speed_kmh and reading.wind_speed_kmh >= 30:
            items.append("rüzgar geçirmez bir dış katman")

        item_list = ", ".join(items)

        template = (
            "[[GREETING]][[LOCATION]] için ([[TEMP_C]]°C, [[CONDITION]]): "
            f"{item_list} önerilir."
        )

        return self._resolve(template, reading, user_name)

    # ------------------------------------------------------------------
    # SAĞLIK ÖNERİSİ
    # ------------------------------------------------------------------
    def _build_health_advice(self, reading: WeatherReading, user_name: Optional[str]) -> str:
        """
        Sıcaklık/rüzgar/yağıştan türetilebilen basit sağlık uyarıları.
        NOT: Gerçek UV indeksi veya polen verisi şu an DI-Agent'ın
        çektiği veride YOK (Open-Meteo'nun temel endpoint'i bunları
        sağlamıyor) - bu yüzden bu MVP'de SADECE sıcaklık/rüzgar
        tabanlı, GERÇEKTEN ELDE OLAN veriye dayalı öneriler üretiliyor.
        UV/polen, ileride DI-Agent'ın Open-Meteo'nun ek parametrelerini
        (uv_index, vb.) çekecek şekilde genişletilmesiyle eklenebilir -
        bu BİLİNÇLİ bir kapsam sınırlaması, "var olmayan veriden öneri
        üretme" riskinden kaçınmak için (whitepaper'ın "context
        hallucination" uyarısıyla aynı mantık).
        """
        warnings = []

        if reading.temperature_c >= 32:
            warnings.append("aşırı sıcak, bol su tüketin ve doğrudan güneşten kaçının")
        if reading.temperature_c <= -5:
            warnings.append("aşırı soğuk, dış mekanda uzun süre kalmayın")
        if reading.wind_speed_kmh and reading.wind_speed_kmh >= 50:
            warnings.append("şiddetli rüzgar, açık alanlarda dikkatli olun")

        if not warnings:
            warning_text = "özel bir sağlık riski görünmüyor"
        else:
            warning_text = ", ".join(warnings)

        template = f"[[GREETING]]Sağlık notu: {warning_text}."
        return self._resolve(template, reading, user_name)

    # ------------------------------------------------------------------
    # AKTİVİTE ÖNERİSİ
    # ------------------------------------------------------------------
    def _build_activity_advice(self, reading: WeatherReading, user_name: Optional[str]) -> str:
        """
        Hava durumuna göre kaba bir "dış mekan uygun mu" değerlendirmesi.
        """
        if reading.condition == "storm" or (reading.precipitation_mm and reading.precipitation_mm > 5):
            suggestion = "dış mekan aktiviteleri yerine kapalı alan planlamanız önerilir"
        elif reading.condition == "rain" or (reading.precipitation_mm and reading.precipitation_mm > 0):
            suggestion = "kısa dış mekan aktiviteleri için şemsiye ile çıkabilirsiniz"
        elif reading.condition == "clear" and 15 <= reading.temperature_c <= 27:
            suggestion = "dış mekan aktiviteleri (yürüyüş, piknik) için ideal bir gün"
        else:
            suggestion = "dış mekan aktiviteleri için durumu kontrol ederek planlayın"

        template = f"[[GREETING]]Aktivite önerisi: {suggestion}."
        return self._resolve(template, reading, user_name)

    # ------------------------------------------------------------------
    # ORTAK YARDIMCI: context_resolver.py'yi çağıran tek nokta
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve(template: str, reading: WeatherReading, user_name: Optional[str]) -> str:
        """
        Şablondaki [[GREETING]], [[LOCATION]], [[TEMP_C]], [[CONDITION]]
        placeholder'larını resolve_context ile gerçek değerlerle değiştirir.

        [[GREETING]] özel bir alan: user_name verilmişse "Ahmet, " gibi
        bir selamlama ekler, verilmemişse BOŞ string olur (yani şablon
        kişiselleştirme olmadan da düzgün çalışır - resolve_context'in
        "değer yoksa placeholder'ı olduğu gibi bırak" davranışına
        GÜVENMİYORUZ burada, çünkü kullanıcıya "[[GREETING]]" yazısının
        sızması KÖTÜ bir deneyim olurdu - bu yüzden override_state'te
        HER ZAMAN bir GREETING değeri sağlanıyor, boş string dahi olsa).
        """
        greeting = f"{user_name}, " if user_name else ""
        override_state = {
            "GREETING": greeting,
            "LOCATION": reading.location,
            "TEMP_C": str(reading.temperature_c),
            "CONDITION": reading.condition,
        }
        return resolve_context(template, override_state=override_state)


if __name__ == "__main__":
    from app.core.models import DataSourceStatus

    agent = PAAgent()

    print("=== TEST 1: Soğuk, yağmurlu, rüzgarlı (İzmir, kişiselleştirilmemiş) ===")
    cold_rainy = WeatherReading(
        location="İzmir", latitude=38.42, longitude=27.14,
        temperature_c=8.0, condition="rain",
        precipitation_mm=3.5, wind_speed_kmh=35.0,
        source="open-meteo", source_status=DataSourceStatus.HEALTHY,
        confidence_score=1.0,
    )
    result1 = agent.advise(cold_rainy)
    print(f"Kıyafet: {result1.payload['clothing_advice']}")
    print(f"Sağlık: {result1.payload['health_advice']}")
    print(f"Aktivite: {result1.payload['activity_advice']}")

    print("\n=== TEST 2: Sıcak, açık hava (Bodrum, KİŞİSELLEŞTİRİLMİŞ - user_name verildi) ===")
    hot_clear = WeatherReading(
        location="Bodrum", latitude=37.04, longitude=27.43,
        temperature_c=34.0, condition="clear",
        precipitation_mm=0.0, wind_speed_kmh=10.0,
        source="open-meteo", source_status=DataSourceStatus.HEALTHY,
        confidence_score=1.0,
    )
    result2 = agent.advise(hot_clear, user_name="Levent")
    print(f"Kıyafet: {result2.payload['clothing_advice']}")
    print(f"Sağlık: {result2.payload['health_advice']}")
    print(f"Aktivite: {result2.payload['activity_advice']}")
    print(">>> Dikkat: Bu öneriler 'Levent, ...' ile başlamalı (context_resolver.py")
    print(">>> ile [[GREETING]] placeholder'ı çözüldü).")

    print("\n=== TEST 3: Ilıman, açık hava - dış mekan için ideal (kişiselleştirilmemiş) ===")
    mild_clear = WeatherReading(
        location="İstanbul", latitude=41.01, longitude=28.98,
        temperature_c=22.0, condition="clear",
        precipitation_mm=0.0, wind_speed_kmh=12.0,
        source="open-meteo", source_status=DataSourceStatus.HEALTHY,
        confidence_score=1.0,
    )
    result3 = agent.advise(mild_clear)
    print(f"Kıyafet: {result3.payload['clothing_advice']}")
    print(f"Sağlık: {result3.payload['health_advice']}")
    print(f"Aktivite: {result3.payload['activity_advice']}")

    print("\n=== TEST 4: Düşük güvenilirlikli veri (LLM-Agent tahmini) - öneri ÜRETİLİR ama uyarı eklenir ===")
    low_confidence = WeatherReading(
        location="Ankara", latitude=39.93, longitude=32.86,
        temperature_c=15.0, condition="unknown",
        source="llm-agent-seasonal-estimate", source_status=DataSourceStatus.CRITICAL,
        confidence_score=0.2,
    )
    result4 = agent.advise(low_confidence)
    print(f"Kıyafet: {result4.payload['clothing_advice']}")
    print(f"{result4.error_message}")
    print(">>> Dikkat: AS-Agent'taki 'düşük güvenle otomatik alert YASAK' kuralı")
    print(">>> burada GEÇERLİ DEĞİL - öneri yine üretiliyor, sadece şeffaflık için")
    print(">>> kaynağın güvenilirliği ayrıca belirtiliyor (zarar seviyesi farklı).")
