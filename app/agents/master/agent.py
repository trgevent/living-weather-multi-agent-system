"""
app/agents/master/agent.py
=============================
Master Agent (Orchestrator)

GÖREV:
  Living Weather'ın "beyni". Kullanıcıdan gelen bir konum sorgusunu
  alır, DI-Agent / LLM-Agent / AS-Agent / FL-Agent'ı doğru sırayla
  çağırır, sonuçları WeatherContext ("paylaşılan kara tahta") üzerinde
  birleştirir ve kullanıcıya tek, anlamlı bir cevap döner.

AKIŞ (whitepaper'ın "Zarif Bozunma" felsefesinin uçtan uca kanıtı):

  1. DI-Agent çağrılır (gerçek API + devre kesici)
     |
     v
  2. DI-Agent'ın confidence_score'u yeterli mi? (>= 0.5)
     |                                   |
     HAYIR                              EVET
     |                                   |
     v                                   v
  3a. LLM-Agent çağrılır              3b. DI-Agent'ın sonucu kullanılır
      (mevsimsel fallback)                (gerçek veri)
     |                                   |
     +-----------------+-----------------+
                       |
                       v
  4. Elde edilen WeatherReading -> AS-Agent'a gönderilir
     (tehlike kontrolü, Policy Server ile korunan uyarı üretimi)
                       |
                       v
  5. WeatherReading -> FL-Agent'a kaydedilir
     (gelecekte gerçek değerle karşılaştırılmak üzere)
                       |
                       v
  6. WeatherContext tamamlanır, kullanıcıya döner

NEDEN BU SIRALAMA?
  whitepaper'ın "Multi-Agent" felsefesinde her ajan TEK bir
  sorumluluğa sahip olmalı (single responsibility). Master Agent'ın
  TEK işi koordinasyon - kendisi hiçbir veri çekmiyor, hiçbir karar
  mantığı içermiyor (örn. "kaç derece tehlikeli" kararını AS-Agent
  veriyor, Master Agent sadece "şimdi AS-Agent'ı çağır" diyor).
  Bu ayrım, ileride herhangi bir ajanı (örn. DI-Agent'ı başka bir
  API sağlayıcısına çevirmek) diğerlerini bozmadan değiştirebilmeyi
  sağlar.

Kaynak: Living Weather mimarisi (kullanıcı tasarımı), Day 5 whitepaper
        "Zero-Trust Development" ve "Evaluation" prensiplerinin
        uçtan uca entegrasyonu.
"""

import sys
import uuid
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.models import WeatherContext, DataSourceStatus
from app.agents.di_agent.agent import DIAgent
from app.agents.llm_agent.agent import LLMAgent
from app.agents.as_agent.agent import ASAgent
from app.agents.fl_agent.agent import FLAgent
from app.agents.ri_agent.agent import RIAgent
from app.agents.pa_agent.agent import PAAgent
from app.agents.mc_agent.agent import MCAgent


# Basit bir konum->koordinat tablosu (gerçek production'da bir
# geocoding API'sine - örn. Open-Meteo'nun kendi geocoding endpoint'ine -
# bağlanılabilir; şimdilik prototip için sabit tablo kullanıyoruz)
KNOWN_LOCATIONS = {
    "izmir": (38.42, 27.14),
    "istanbul": (41.01, 28.98),
    "ankara": (39.93, 32.86),
    "antalya": (36.90, 30.70),
    "bodrum": (37.04, 27.43),
}


class MasterAgent:
    """
    Living Weather'ın orkestra şefi. Diğer 4 çekirdek ajanı koordine eder.
    """

    def __init__(self):
        self.agent_name = "Master-Agent"
        self.di_agent = DIAgent()
        self.llm_agent = LLMAgent()
        self.as_agent = ASAgent()
        self.fl_agent = FLAgent()
        # RI-Agent (ESNEK AJAN, Gün 3'te eklendi): aynı DIAgent örneğini
        # paylaşır - böylece RI-Agent'ın çağırdığı DI-Agent isteği de
        # AYNI devre kesici (Redis) state'ini kullanır, ayrı bir devre
        # kesici "kör nokta" oluşmaz.
        self.ri_agent = RIAgent(di_agent=self.di_agent)
        # PA-Agent (ESNEK AJAN, Gün 5'te eklendi): bağımsız, dış API/LLM
        # çağrısı gerektirmiyor, sadece DI-Agent'ın çıktısını işliyor -
        # bu yüzden ayrı bir dependency injection'a gerek yok.
        self.pa_agent = PAAgent()
        # MC-Agent (ESNEK AJAN, Gün 6'da eklendi - esnek 3'ün SONUNCUSU):
        # RI-Agent'taki AYNI desen - aynı DIAgent örneğini paylaşır,
        # ayrı bir devre kesici "kör nokta" oluşmaz. KNOWN_LOCATIONS
        # tablosuna BAĞIMLI DEĞİL (bkz. CHAPTER_NOTES.md, "MC-Agent
        # serbest koordinat çiftleriyle çalışır" kararı) - Master Agent
        # üzerinden çağrıldığında konum adı çözümü için bu tabloyu
        # kullanabilir, ama MC-Agent'ın KENDİSİ tabloyu hiç bilmez.
        self.mc_agent = MCAgent(di_agent=self.di_agent)

    def process_request(self, location_query: str) -> WeatherContext:
        """
        Ana giriş noktası: bir konum sorgusunu uçtan uca işler.
        """
        request_id = str(uuid.uuid4())[:8]
        context = WeatherContext(request_id=request_id, location_query=location_query)

        # 1. Koordinatları çöz
        coords = self._resolve_coordinates(location_query)
        if coords is None:
            context.overall_status = DataSourceStatus.UNKNOWN
            return context
        context.latitude, context.longitude = coords

        # 2. DI-Agent'ı çağır
        di_response = self.di_agent.fetch(
            latitude=context.latitude,
            longitude=context.longitude,
            location_name=location_query,
        )
        context.agent_trace.append(di_response)

        # 3. Güven skoruna göre LLM-Agent'a düş ya da DI-Agent sonucunu kullan
        di_confidence = di_response.confidence_score
        if self.llm_agent.should_activate(di_confidence):
            llm_response = self.llm_agent.estimate(
                latitude=context.latitude,
                longitude=context.longitude,
                location_name=location_query,
            )
            context.agent_trace.append(llm_response)
            active_reading_data = llm_response.payload["reading"]
            context.overall_status = DataSourceStatus.CRITICAL
        else:
            active_reading_data = di_response.payload["reading"]
            context.overall_status = DataSourceStatus.HEALTHY

        from app.core.models import WeatherReading
        current_reading = WeatherReading(**active_reading_data)
        context.current_reading = current_reading

        # 4. AS-Agent'a gönder (tehlike kontrolü)
        as_response = self.as_agent.evaluate(current_reading)
        context.agent_trace.append(as_response)
        if as_response.payload.get("alert"):
            from app.core.models import WeatherAlert
            context.alerts.append(WeatherAlert(**as_response.payload["alert"]))

        # 5. FL-Agent'a kaydet (gelecekteki değerlendirme için)
        self.fl_agent.record_prediction(current_reading)

        return context

    def process_route_request(self, origin_query: str, destination_query: str):
        """
        ESNEK AJAN GİRİŞ NOKTASI (Gün 3'te eklendi): RI-Agent'ı çağırır.
        process_request()'TEN AYRI - mevcut akışa hiç dokunmadan eklendi,
        çünkü RI-Agent çekirdek 5'in parçası DEĞİL (MASTERPLAN.md'deki
        "esnek 3 ajan" listesinden). Konum çözümü process_request'teki
        AYNI _resolve_coordinates() metodunu kullanıyor (DRY).

        Döner: (origin_coords, destination_coords, AgentResponse) ya da
        konumlardan biri tanınmazsa (None, None, None).
        """
        origin_coords = self._resolve_coordinates(origin_query)
        destination_coords = self._resolve_coordinates(destination_query)

        if origin_coords is None or destination_coords is None:
            return None, None, None

        route_response = self.ri_agent.plan_route(
            origin=origin_query, origin_coords=origin_coords,
            destination=destination_query, destination_coords=destination_coords,
        )
        return origin_coords, destination_coords, route_response

    def process_advisory_request(self, location_query: str, user_name: Optional[str] = None):
        """
        ESNEK AJAN GİRİŞ NOKTASI (Gün 5'te eklendi): PA-Agent'ı çağırır.
        process_request()'TEN AYRI - mevcut akışa hiç dokunmadan eklendi,
        çünkü PA-Agent da çekirdek 5'in parçası DEĞİL.

        Akış: önce process_request() ile normal hava verisi akışı
        (DI-Agent → gerekirse LLM-Agent → AS-Agent) çalıştırılır - bu
        sayede PA-Agent, AS-Agent'ın ZATEN ürettiği WeatherContext'i
        yeniden kullanır (DRY, ayrı bir DI-Agent çağrısı tekrar
        yapılmaz). Sonra context.current_reading, PA-Agent'a verilir.

        Döner: (WeatherContext, AgentResponse) ya da konum tanınmazsa
        (WeatherContext, None).
        """
        context = self.process_request(location_query)

        if context.current_reading is None:
            return context, None

        advisory_response = self.pa_agent.advise(context.current_reading, user_name=user_name)
        context.agent_trace.append(advisory_response)
        return context, advisory_response

    def process_microclimate_request(self, point_a_query: str, point_b_query: str):
        """
        ESNEK AJAN GİRİŞ NOKTASI (Gün 6'da eklendi - esnek 3'ün
        SONUNCUSU): MC-Agent'ı çağırır. process_request()'TEN AYRI -
        mevcut akışa hiç dokunmadan eklendi, çünkü MC-Agent da çekirdek
        5'in parçası DEĞİL.

        NOT: Bu giriş noktası, Living Weather'ın KENDİ KNOWN_LOCATIONS
        tablosuyla çalışan bir KOLAYLIK katmanıdır - _resolve_coordinates()
        ile konum adını koordinata çevirir (RI-Agent'taki process_route_
        request() ile AYNI desen). MC-Agent'ın KENDİSİ (mc_agent.agent.
        MCAgent.compare_locations) bu tabloyu hiç bilmez, serbest
        (latitude, longitude) çiftleri alır - bu yüzden ileride Atlas/
        HeyGuide gibi başka projelerden çağrılırken bu metodun (ve
        KNOWN_LOCATIONS tablosunun) ATLANIP doğrudan MCAgent.compare_
        locations() çağrılması yeterlidir (bkz. CHAPTER_NOTES.md).

        Döner: (point_a_coords, point_b_coords, AgentResponse) ya da
        konumlardan biri tanınmazsa (None, None, None).
        """
        point_a_coords = self._resolve_coordinates(point_a_query)
        point_b_coords = self._resolve_coordinates(point_b_query)

        if point_a_coords is None or point_b_coords is None:
            return None, None, None

        comparison_response = self.mc_agent.compare_locations(
            point_a_name=point_a_query, point_a_coords=point_a_coords,
            point_b_name=point_b_query, point_b_coords=point_b_coords,
        )
        return point_a_coords, point_b_coords, comparison_response

    @staticmethod
    def _resolve_coordinates(location_query: str):
        """Basit konum -> koordinat çözümü (Türkçe normalize ile)."""
        # LLMAgent'taki normalize fonksiyonunu yeniden kullanıyoruz
        # (DRY prensibi - aynı mantığı iki yerde yazmamak için)
        normalized = LLMAgent._normalize_turkish(location_query)
        return KNOWN_LOCATIONS.get(normalized)

    def summarize(self, context: WeatherContext) -> str:
        """İnsan-okunur bir özet üretir (CLI/demo amaçlı)."""
        lines = [f"--- Living Weather Raporu ({context.request_id}) ---"]
        lines.append(f"Konum: {context.location_query}")

        if context.current_reading:
            r = context.current_reading
            lines.append(
                f"Sıcaklık: {r.temperature_c}°C | Durum: {r.condition} | "
                f"Kaynak: {r.source} | Güven: {r.confidence_score:.0%}"
            )
        else:
            lines.append("Hava verisi alınamadı (konum tanınmıyor).")

        if context.alerts:
            for alert in context.alerts:
                lines.append(f"⚠️  UYARI [{alert.severity.value}]: {alert.title} - {alert.description}")
        else:
            lines.append("Aktif uyarı yok.")

        lines.append(f"Sistem durumu: {context.overall_status.value}")
        lines.append(f"Ajan izleri: {[t.agent_name for t in context.agent_trace]}")
        return "\n".join(lines)


if __name__ == "__main__":
    master = MasterAgent()

    print("=== SENARYO 1: Bilinen şehir, normal akış (gerçek API beklenir) ===")
    print("NOT: Bu senaryonun sonucu, çalıştığın makinenin internet erişimine göre")
    print("değişir. Gerçek internet erişimi olan bir makinede 'source: open-meteo' ve")
    print("'Sistem durumu: healthy' görmen beklenir. Kısıtlı/sandbox bir ortamda ise")
    print("LLM-Agent fallback'i devreye girebilir (source: llm-agent-seasonal-estimate)")
    print("- bu durumda bile sistem ÇÖKMEDEN, düşük güvenle çalışmayı sürdürür.\n")
    context1 = master.process_request("İzmir")
    print(master.summarize(context1))

    print("\n" + "=" * 60)
    print("=== SENARYO 2: Tanınmayan şehir (Wakanda) - koordinat çözülemez ===")
    context2 = master.process_request("Wakanda")
    print(master.summarize(context2))

    print("\n" + "=" * 60)
    print("=== SENARYO 3: DI-Agent'ı bilerek bozup LLM-Agent fallback'ini zorlama ===")
    master.di_agent._fetch_open_meteo = lambda *a, **k: (_ for _ in ()).throw(
        ConnectionError("Simüle edilmiş kesinti")
    )
    # Devre kesicinin hemen açılması için threshold'u 1'e düşürüyoruz (sadece bu test için)
    master.di_agent.circuit.failure_threshold = 1
    context3 = master.process_request("Ankara")
    print(master.summarize(context3))
    print(">>> Dikkat: source='llm-agent-seasonal-estimate' ve düşük güven skoru görülmeli,")
    print(">>> bu da Master Agent'ın DI-Agent çöktüğünde otomatik olarak LLM-Agent'a")
    print(">>> geçtiğinin (Zarif Bozunma) kanıtıdır.")

    print("\n" + "=" * 60)
    print("=== SENARYO 4: RI-Agent entegrasyonu - İzmir -> Bodrum rota önerisi ===")
    print("NOT: TEMİZ bir MasterAgent örneği kullanılıyor (SENARYO 3'teki")
    print("bozuk DI-Agent state'i bu örneğe miras kalmasın diye).\n")
    fresh_master = MasterAgent()
    origin_coords, destination_coords, route_response = fresh_master.process_route_request("İzmir", "Bodrum")
    if route_response is None:
        print("Konumlardan biri tanınmadı.")
    else:
        print(f"Rota metni kaynağı: {route_response.payload['text_source']}")
        print(f"Rota metni: {route_response.payload['route_text']}")
        print(f"Judge skoru: {route_response.payload['judge_score']} / 5.0")
        print(f"Genel güven: {route_response.confidence_score:.2f}")
        print(">>> RI-Agent, Master Agent üzerinden çağrıldı, aynı DI-Agent")
        print(">>> (ve aynı Redis devre kesici state'i) paylaşıldı.")

    print("\n" + "=" * 60)
    print("=== SENARYO 5: PA-Agent entegrasyonu - İzmir için kişisel öneri ===")
    print("NOT: TEMİZ bir MasterAgent örneği kullanılıyor.\n")
    fresh_master2 = MasterAgent()
    context5, advisory_response = fresh_master2.process_advisory_request("İzmir", user_name="Levent")
    if advisory_response is None:
        print("Konum tanınmadı, öneri üretilemedi.")
    else:
        print(f"Kıyafet önerisi: {advisory_response.payload['clothing_advice']}")
        print(f"Sağlık önerisi: {advisory_response.payload['health_advice']}")
        print(f"Aktivite önerisi: {advisory_response.payload['activity_advice']}")
        print(f"Ajan izleri: {[t.agent_name for t in context5.agent_trace]}")
        print(">>> PA-Agent, Master Agent üzerinden çağrıldı, process_request()'in")
        print(">>> ZATEN ürettiği WeatherContext'i yeniden kullandı (ayrı bir")
        print(">>> DI-Agent çağrısı tekrar yapılmadı - context.agent_trace'te")
        print(">>> DI-Agent/AS-Agent + PA-Agent'ın hepsi görünmeli).")

    print("\n" + "=" * 60)
    print("=== SENARYO 6: MC-Agent entegrasyonu - İzmir vs Ankara mikro-iklim ===")
    print("NOT: TEMİZ bir MasterAgent örneği kullanılıyor.\n")
    fresh_master3 = MasterAgent()
    pa_coords, pb_coords, comparison_response = fresh_master3.process_microclimate_request("İzmir", "Ankara")
    if comparison_response is None:
        print("Konumlardan biri tanınmadı.")
    else:
        print(f"Karşılaştırma metni kaynağı: {comparison_response.payload['text_source']}")
        print(f"Karşılaştırma: {comparison_response.payload['comparison_text']}")
        print(f"Sıcaklık farkı (A-B): {comparison_response.payload['temperature_diff_c']}°C")
        print(f"Judge skoru: {comparison_response.payload['judge_score']} / 5.0")
        print(f"Genel güven: {comparison_response.confidence_score:.2f}")
        print(">>> MC-Agent, Master Agent üzerinden çağrıldı, aynı DI-Agent")
        print(">>> (ve aynı Redis devre kesici state'i) paylaşıldı - RI-Agent'taki")
        print(">>> AYNI dependency-injection deseni.")
