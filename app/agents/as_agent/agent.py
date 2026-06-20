"""
app/agents/as_agent/agent.py
==============================
AS-Agent (Alert & Safety Agent)

GÖREV:
  Şiddetli hava koşullarını tespit edip uyarı üretir. Bu ajan, Day 5
  whitepaper'ının "Zero-Trust Development" ve "Policy Server" pattern'lerini
  DOĞRUDAN kullanır - çünkü bir uyarı sistemi, en çok kötüye kullanıma
  açık (yanlış alarm, spam, agent hallucination) bileşenlerden biridir.

NEDEN POLICY SERVER GEREKLİ?
  Bir uyarı ajanının iki riski var:
    1. YANLIŞ ALARM: Ajan, gerçekte tehlikeli olmayan bir durumu
       "EXTREME" diye işaretleyip gereksiz panik yaratabilir
       (whitepaper'daki "context hallucination" riski - DI-Agent'tan
       gelen düşük güvenilirlikli veriye dayanarak hatalı karar verme)
    2. YETKİ AŞIMI: AS-Agent'ın SADECE uyarı göndermesi gerekiyor,
       başka bir tool'u (örn. send_email, kullanıcı verisi okuma)
       kullanmaya çalışmamalı. Policy Server'daki "as_agent" rolü
       bunu yapısal olarak (YAML ile) garanti eder - LLM'in "iyi
       niyetli" bir hata yapıp başka bir şey denemesi durumunda da
       devre dışı bırakılır.

NASIL ÇALIŞIR?
  1. WeatherReading'i alır, eşik kontrolü yapar (örn: sıcaklık < -10
     veya > 40, ya da rüzgar > 80 km/h gibi "ekstrem" sayılan değerler)
  2. Eşik aşılırsa, ToolPolicyEngine üzerinden "send_severe_weather_alert"
     tool'unu çağırmaya çalışır
  3. ÖNEMLİ: düşük güvenilirlikli veriden (confidence_score < 0.5)
     gelen okumalar için uyarı üretmez - whitepaper'ın "yanlış alarm"
     riskini azaltma stratejisi budur. Düşük güvenilirlikli veri
     "belki tehlikeli" der ama insan onayı ister (human-in-the-loop),
     otomatik uyarı GÖNDERMEZ.

Kaynak: Living Weather mimarisi + Day 5 whitepaper "Policy Server" ve
        "Zero-Trust Development: Building the Safety Net" bölümleri,
        bu projenin app/core/policy_server.py ve tool_policy_engine.py
        dosyaları üzerine inşa edilmiştir.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))

from app.core.models import WeatherReading, WeatherAlert, AlertSeverity, AgentResponse
from app.core.policy_server import PolicyService


# Uyarı eşikleri - bu değerler whitepaper'ın "Implementing Guardrails"
# bölümündeki "risk profili" kavramına denk geliyor.
ALERT_THRESHOLDS = {
    "extreme_heat_c": 40.0,
    "extreme_cold_c": -10.0,
    "severe_wind_kmh": 80.0,
    "heavy_rain_mm": 50.0,
    "min_confidence_for_auto_alert": 0.5,  # bunun altı için otomatik uyarı YOK
}


class ASAgent:
    """
    Hava verisini değerlendirip, gerektiğinde Policy Server'dan
    geçirerek uyarı üreten ajan.
    """

    def __init__(self, policy_file: str = None):
        if policy_file is None:
            policy_file = str(Path(__file__).resolve().parents[2] / "core" / "policies.yaml")
        # AS-Agent, "as_agent" rolüyle çalışır - sadece uyarı tool'unu kullanabilir
        self.policy = PolicyService(role="as_agent", env="production", policy_file=policy_file)
        self.agent_name = "AS-Agent"

    def evaluate(self, reading: WeatherReading) -> AgentResponse:
        """
        Bir hava okumasını değerlendirir, gerekirse uyarı üretir.
        """
        alert = self._check_thresholds(reading)

        if alert is None:
            return AgentResponse(
                agent_name=self.agent_name,
                success=True,
                payload={"alert_triggered": False},
                confidence_score=1.0,
            )

        # Düşük güvenilirlikli veri için otomatik uyarı GÖNDERME,
        # sadece "human review gerekli" işaretiyle döndür.
        if reading.confidence_score < ALERT_THRESHOLDS["min_confidence_for_auto_alert"]:
            alert.requires_human_review = True
            return AgentResponse(
                agent_name=self.agent_name,
                success=True,
                payload={
                    "alert_triggered": False,
                    "pending_human_review": alert.model_dump(mode="json"),
                },
                confidence_score=reading.confidence_score,
                error_message=(
                    f"Olası tehlike tespit edildi AMA veri güvenilirliği düşük "
                    f"({reading.confidence_score}), otomatik uyarı GÖNDERİLMEDİ. "
                    f"İnsan onayı gerekiyor."
                ),
            )

        # Yeterince güvenilir veri - Policy Server üzerinden uyarı göndermeyi dene
        decision = self.policy.authorize(
            tool_name="send_severe_weather_alert",
            action_description=(
                f"Şiddetli hava uyarısı gönderiliyor: {alert.title} - "
                f"konum: {alert.location}, sebep: {alert.triggered_by}"
            ),
        )

        if not decision["allowed"]:
            return AgentResponse(
                agent_name=self.agent_name,
                success=False,
                payload={"alert_triggered": False, "blocked_alert": alert.model_dump(mode="json")},
                confidence_score=reading.confidence_score,
                error_message=f"Policy tarafından reddedildi: layer={decision['layer']}, "
                              f"reason={decision['reason']}",
            )

        # Her şey geçti - uyarı "gönderildi" (gerçek SMS/push entegrasyonu
        # ileride buraya eklenecek, şimdilik tetikleme mantığını kanıtlıyoruz)
        return AgentResponse(
            agent_name=self.agent_name,
            success=True,
            payload={"alert_triggered": True, "alert": alert.model_dump(mode="json")},
            confidence_score=reading.confidence_score,
        )

    @staticmethod
    def _check_thresholds(reading: WeatherReading) -> WeatherAlert | None:
        """Hava verisinin tehlikeli eşikleri aşıp aşmadığını kontrol eder."""
        if reading.temperature_c >= ALERT_THRESHOLDS["extreme_heat_c"]:
            return WeatherAlert(
                severity=AlertSeverity.EXTREME,
                title="Aşırı Sıcaklık Uyarısı",
                description=f"{reading.location}'da sıcaklık {reading.temperature_c}°C'ye ulaştı.",
                location=reading.location,
                triggered_by=f"temperature_c >= {ALERT_THRESHOLDS['extreme_heat_c']}",
            )
        if reading.temperature_c <= ALERT_THRESHOLDS["extreme_cold_c"]:
            return WeatherAlert(
                severity=AlertSeverity.EXTREME,
                title="Aşırı Soğuk Uyarısı",
                description=f"{reading.location}'da sıcaklık {reading.temperature_c}°C'ye düştü.",
                location=reading.location,
                triggered_by=f"temperature_c <= {ALERT_THRESHOLDS['extreme_cold_c']}",
            )
        if reading.wind_speed_kmh and reading.wind_speed_kmh >= ALERT_THRESHOLDS["severe_wind_kmh"]:
            return WeatherAlert(
                severity=AlertSeverity.SEVERE,
                title="Şiddetli Rüzgar Uyarısı",
                description=f"{reading.location}'da rüzgar hızı {reading.wind_speed_kmh} km/h.",
                location=reading.location,
                triggered_by=f"wind_speed_kmh >= {ALERT_THRESHOLDS['severe_wind_kmh']}",
            )
        if reading.precipitation_mm and reading.precipitation_mm >= ALERT_THRESHOLDS["heavy_rain_mm"]:
            return WeatherAlert(
                severity=AlertSeverity.WARNING,
                title="Yoğun Yağış Uyarısı",
                description=f"{reading.location}'da yağış {reading.precipitation_mm} mm.",
                location=reading.location,
                triggered_by=f"precipitation_mm >= {ALERT_THRESHOLDS['heavy_rain_mm']}",
            )
        return None


if __name__ == "__main__":
    from app.core.models import DataSourceStatus

    agent = ASAgent()

    print("=== TEST 1: Normal hava, uyarı YOK bekleniyor ===")
    normal_reading = WeatherReading(
        location="İzmir", latitude=38.42, longitude=27.14,
        temperature_c=24.0, wind_speed_kmh=15.0,
        source="open-meteo", source_status=DataSourceStatus.HEALTHY,
        confidence_score=1.0,
    )
    result1 = agent.evaluate(normal_reading)
    print(f"Uyarı tetiklendi mi: {result1.payload.get('alert_triggered')}")

    print("\n=== TEST 2: Aşırı sıcaklık, GÜVENİLİR veri -> uyarı bekleniyor ===")
    hot_reading = WeatherReading(
        location="İzmir", latitude=38.42, longitude=27.14,
        temperature_c=42.0,
        source="open-meteo", source_status=DataSourceStatus.HEALTHY,
        confidence_score=1.0,
    )
    result2 = agent.evaluate(hot_reading)
    print(f"Uyarı tetiklendi mi: {result2.payload.get('alert_triggered')}")
    if result2.payload.get("alert"):
        print(f"Uyarı detayı: {result2.payload['alert']['title']} - {result2.payload['alert']['severity']}")

    print("\n=== TEST 3: Aşırı sıcaklık AMA DÜŞÜK GÜVENİLİR veri -> insan onayı bekleniyor (otomatik uyarı YOK) ===")
    unreliable_hot_reading = WeatherReading(
        location="İzmir", latitude=38.42, longitude=27.14,
        temperature_c=45.0,
        source="llm-agent-seasonal-estimate", source_status=DataSourceStatus.CRITICAL,
        confidence_score=0.2,  # düşük güven
    )
    result3 = agent.evaluate(unreliable_hot_reading)
    print(f"Uyarı tetiklendi mi: {result3.payload.get('alert_triggered')}")
    print(f"İnsan onayı bekleyen var mı: {result3.payload.get('pending_human_review') is not None}")
    print(f"Not: {result3.error_message}")
