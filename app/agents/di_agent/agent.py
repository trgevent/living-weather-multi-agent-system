"""
app/agents/di_agent/agent.py
==============================
DI-Agent (Veri Alımı / Data Ingestion Agent)

GÖREV:
  Gerçek hava durumu verisini dış API'lerden çeker. "Zarif Bozunma"
  felsefesinin İLK katmanı: birincil kaynak (Open-Meteo) çökerse,
  devre kesici (circuit breaker) devreye girer ve yedek kaynağa
  (bu prototipte basitleştirilmiş bir fallback) geçilir.

NEDEN DEVRE KESİCİ (CIRCUIT BREAKER)?
  Eğer her hava durumu isteğinde Open-Meteo'ya bağlanmaya çalışıp
  her seferinde 10 saniye timeout beklersek, API çökmüşken sistem
  tamamen yavaşlar/kilitlenir. Devre kesici mantığı: "Open-Meteo
  art arda N kere başarısız oldu mu? Olduysa, bir süre (cooldown)
  hiç denemeden direkt yedek kaynağa git, sistemi yorma."

  3 durum (state):
    CLOSED   -> Her şey normal, Open-Meteo'ya istek gönderiliyor
    OPEN     -> Open-Meteo'ya art arda çok hata aldık, şimdilik
                hiç deneme, direkt fallback kullan
    HALF_OPEN-> Cooldown süresi geçti, "bir kere deneyelim, düzeldi mi"

GERÇEK API KULLANIMI:
  Open-Meteo TAMAMEN ÜCRETSİZ ve API KEY GEREKTİRMİYOR
  (https://open-meteo.com/) - bu yüzden billing/quota derdi yok,
  Living Weather'ın bu ajanı şimdiden gerçek veriyle çalışabilir.

Kaynak: Living Weather mimarisi (kullanıcı tasarımı) +
        Day 5 whitepaper "Zero-Trust Development" felsefesi (devre
        kesici kavramı, sandbox/guardrail mantığının veri katmanına
        uygulanmış hali)
"""

import time
import httpx
from enum import Enum
from typing import Optional
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.models import WeatherReading, DataSourceStatus, AgentResponse
from app.core.redis_circuit_breaker import RedisCircuitBreaker, get_redis_client


class CircuitState(str, Enum):
    CLOSED = "closed"        # normal çalışma
    OPEN = "open"            # devre açık, fallback kullan
    HALF_OPEN = "half_open"  # test ediliyor


class CircuitBreaker:
    """
    Basit, bağımsız bir devre kesici implementasyonu.
    Gerçek production'da Redis'te durum tutulur (birden fazla worker
    aynı devre durumunu paylaşsın diye); bu prototipte bellek-içi
    (in-memory) tutuyoruz, kolayca Redis'e taşınabilir.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at: Optional[float] = None

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.opened_at = time.monotonic()

    def can_attempt(self) -> bool:
        """Şu an birincil kaynağı denemeye değer mi?"""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            assert self.opened_at is not None
            if time.monotonic() - self.opened_at >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN: bir şans ver
        return True


class DIAgent:
    """
    Hava durumu verisini dış kaynaklardan çeken ajan.
    Devre kesici ile korunan birincil kaynak (Open-Meteo).
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 30.0):
        """
        DEĞİŞİKLİK (Redis entegrasyonu, 20 Haziran 2026):
        Eskiden burada bellek-içi CircuitBreaker(...) oluşturuluyordu
        (yukarıdaki CircuitBreaker sınıfı - referans için silinmedi,
        ama artık AKTİF OLARAK KULLANILMIYOR). Şimdi RedisCircuitBreaker
        kullanıyoruz - state, bu Python process'i yeniden başlasa,
        ya da FastAPI birden fazla worker ile çalışsa bile PAYLAŞILIYOR
        (bkz. app/core/redis_circuit_breaker.py docstring'i, STATUS.md'de
        not edilen "her worker kendi devre durumunu tutar" riskinin
        çözümü budur).

        Public arayüz (can_attempt, record_success, record_failure)
        AYNI KALDI - bu yüzden aşağıdaki fetch() metodunda HİÇBİR
        SATIR DEĞİŞMEDİ.
        """
        redis_client = get_redis_client()
        self.circuit = RedisCircuitBreaker(
            redis_client,
            key_prefix="circuit:di_agent:open-meteo",
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
        )
        self.agent_name = "DI-Agent"

    def fetch(self, latitude: float, longitude: float, location_name: str) -> AgentResponse:
        """
        Ana giriş noktası. Devre kesici durumuna göre Open-Meteo'yu
        dener veya direkt fallback'e geçer.
        """
        start = time.monotonic()

        if self.circuit.can_attempt():
            try:
                reading = self._fetch_open_meteo(latitude, longitude, location_name)
                self.circuit.record_success()
                elapsed_ms = (time.monotonic() - start) * 1000
                return AgentResponse(
                    agent_name=self.agent_name,
                    success=True,
                    payload={"reading": reading.model_dump(mode="json")},
                    confidence_score=1.0,
                    processing_time_ms=elapsed_ms,
                )
            except Exception as exc:
                self.circuit.record_failure()
                # Birincil kaynak çöktü, hemen fallback'e düş
                return self._fallback(latitude, longitude, location_name, start, str(exc))
        else:
            # Devre açık, hiç denemeden fallback'e git
            return self._fallback(
                latitude, longitude, location_name, start,
                reason=f"Circuit breaker OPEN (cooldown: {self.circuit.cooldown_seconds}s)",
            )

    def _fetch_open_meteo(
        self, latitude: float, longitude: float, location_name: str
    ) -> WeatherReading:
        """
        Open-Meteo'dan GERÇEK veri çeker. API key gerektirmez, ücretsizdir.
        """
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "timeout": 8,
        }
        response = httpx.get(url, params=params, timeout=8.0)
        response.raise_for_status()
        data = response.json()
        current = data["current"]

        return WeatherReading(
            location=location_name,
            latitude=latitude,
            longitude=longitude,
            temperature_c=current["temperature_2m"],
            humidity_pct=current.get("relative_humidity_2m"),
            wind_speed_kmh=current.get("wind_speed_10m"),
            precipitation_mm=current.get("precipitation"),
            condition=self._infer_condition(current.get("precipitation", 0)),
            source="open-meteo",
            source_status=DataSourceStatus.HEALTHY,
            confidence_score=1.0,
        )

    @staticmethod
    def _infer_condition(precipitation: float) -> str:
        if precipitation and precipitation > 5:
            return "storm"
        if precipitation and precipitation > 0:
            return "rain"
        return "clear"

    def _fallback(
        self,
        latitude: float,
        longitude: float,
        location_name: str,
        start_time: float,
        reason: str,
    ) -> AgentResponse:
        """
        Birincil kaynak başarısızsa devreye giren basitleştirilmiş
        yedek mantık. Gerçek production'da burada VisualCrossing/
        NOAA gibi ikinci bir API olurdu; bu prototipte LLM-Agent'a
        devretmeden önceki son "sentetik tahmin" katmanı olarak
        basit bir iklim ortalaması döndürüyoruz (düşük güven skoruyla
        işaretli, bu önemli - kullanıcı bunun "tahmini" olduğunu bilsin).
        """
        elapsed_ms = (time.monotonic() - start_time) * 1000
        fallback_reading = WeatherReading(
            location=location_name,
            latitude=latitude,
            longitude=longitude,
            temperature_c=20.0,  # kaba mevsimsel ortalama, gerçek değil
            condition="unknown",
            source="fallback-degraded",
            source_status=DataSourceStatus.DEGRADED,
            confidence_score=0.3,  # düşük güven - bu gerçek veri değil
        )
        return AgentResponse(
            agent_name=self.agent_name,
            success=True,  # sistem çökmedi, sadece düşük güvenle çalışıyor
            payload={"reading": fallback_reading.model_dump(mode="json")},
            confidence_score=0.3,
            error_message=f"Birincil kaynak başarısız, fallback kullanıldı: {reason}",
            processing_time_ms=elapsed_ms,
        )


if __name__ == "__main__":
    print("=== TEST 1: Gerçek Open-Meteo isteği (İzmir) ===")
    agent = DIAgent()
    result = agent.fetch(latitude=38.42, longitude=27.14, location_name="İzmir")
    print(f"Başarılı: {result.success}")
    print(f"Güven skoru: {result.confidence_score}")
    print(f"İşlem süresi: {result.processing_time_ms:.1f} ms")
    print(f"Veri: {result.payload}")

    print("\n=== TEST 2: Devre kesici simülasyonu (sahte hata ile) ===")
    print("NOT: Bu testin Redis key'i ('circuit:di_agent:open-meteo'), TEST 1'in")
    print("kullandığı agent ile AYNI prefix'i paylaşıyor olabilir eğer aynı")
    print("failure_threshold/cooldown ile çağrılsaydı - state artık Redis'te")
    print("yaşadığı için, temiz bir başlangıç için key'i elle siliyoruz.")
    from app.core.redis_circuit_breaker import get_redis_client as _get_client
    _get_client().delete("circuit:di_agent:open-meteo")

    broken_agent = DIAgent(failure_threshold=2, cooldown_seconds=5.0)
    # _fetch_open_meteo'yu bilerek bozuk bir URL'e yönlendirerek hata simüle ediyoruz
    broken_agent._fetch_open_meteo = lambda *a, **k: (_ for _ in ()).throw(
        ConnectionError("Simüle edilmiş ağ hatası")
    )
    for i in range(3):
        r = broken_agent.fetch(latitude=38.42, longitude=27.14, location_name="İzmir")
        print(
            f"  Deneme {i+1}: success={r.success}, confidence={r.confidence_score}, "
            f"circuit_state={broken_agent.circuit.state.value}"
        )
    # Beklenen: ilk 2 deneme "failure" sayılır (threshold=2), 3. denemede
    # devre OPEN olmuş olmalı ve fallback'e hiç denemeden gidilmeli
