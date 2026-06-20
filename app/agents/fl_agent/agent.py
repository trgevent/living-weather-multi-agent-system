"""
app/agents/fl_agent/agent.py
==============================
FL-Agent (Feedback & Learning Agent)

GÖREV:
  Sistemin geçmiş tahminlerini gerçekleşen hava durumuyla karşılaştırır,
  hangi veri kaynağının (DI-Agent / Open-Meteo, LLM-Agent / mevsimsel
  tahmin) ne kadar güvenilir olduğunu ölçer ve raporlar. Bu, Day 5
  whitepaper'ın "Evaluation" pattern'inin (binary test DEĞİL, tolerans
  bandı + skor) DOĞRUDAN uygulamasıdır.

NEDEN BİNARY TEST YETERSİZ?
  "Tahmin == Gerçek mi?" diye sormak hava durumu gibi olasılıksal bir
  alanda anlamsız - %100 isabet beklenemez. FL-Agent bunun yerine:
    - Sayısal tahminler (sıcaklık) için TOLERANS BANDI kullanır
      (evaluation_engine.py / evaluate_numeric_tolerance)
    - Kaynağa göre AYRI istatistik tutar - yani "Open-Meteo'nun
      ortalama hata payı kaç derece, LLM-Agent'ınki kaç derece"
      diye karşılaştırma yapabilir (bu, hangi kaynağa ne kadar
      güvenilmesi gerektiğine dair gerçek veriye dayalı bir karar
      üretir - "model kalibrasyonu" budur)

GERÇEK HAYATTA NASIL ÇALIŞIR?
  1. DI-Agent veya LLM-Agent bir tahmin üretir, bu tahmin kaydedilir
     (FeedbackRecord olarak, henüz "actual" alanı boş)
  2. Birkaç saat/gün sonra, gerçek hava durumu öğrenilir (örn. bir
     sonraki DI-Agent çağrısının "current" okumasından, ya da
     kullanıcı geri bildiriminden)
  3. FL-Agent, prediction ile actual'ı karşılaştırır, tolerans bandı
     testi uygular, sonucu kaydeder
  4. Zamanla, "kaynak X'in ortalama skoru Y" gibi bir rapor üretilebilir
     - bu da Master Agent'ın hangi kaynağa ne kadar ağırlık vereceğine
       (weighted consensus) karar vermesinde kullanılabilir

Kaynak: Living Weather mimarisi + Day 5 whitepaper "Evaluation" bölümü
        (s.29-30), bu projenin app/core/evaluation_engine.py dosyası
        üzerine inşa edilmiştir.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.core.models import FeedbackRecord, WeatherReading, AgentResponse
from app.core.evaluation_engine import evaluate_numeric_tolerance, EvalReport
from app.core.db import FeedbackRepository


# Kaynak bazlı tolerans bantları - her veri kaynağının doğası farklı,
# bu yüzden aynı tolerans bandını hepsine uygulamak adil olmaz.
TOLERANCE_BY_SOURCE = {
    "open-meteo": 2.0,                    # gerçek API, sıkı tolerans
    "fallback-degraded": 5.0,             # yedek kaynak, daha gevşek
    "llm-agent-seasonal-estimate": 8.0,   # mevsimsel tahmin, en gevşek
}
DEFAULT_TOLERANCE = 5.0


class FLAgent:
    """
    Geçmiş tahminleri gerçek değerlerle karşılaştırıp kalibrasyon
    raporu üreten ajan.
    """

    def __init__(self, repository: FeedbackRepository | None = None):
        """
        DEĞİŞİKLİK (Redis/PostgreSQL entegrasyonu, 20 Haziran 2026):
        self._history (bellek-içi liste) KALDIRILDI - artık PostgreSQL'e
        yazıyoruz. repository parametresi opsiyonel: testlerde farklı
        bir DSN ile (örn. test veritabanı) kolayca değiştirilebilsin
        diye dependency injection olarak verildi. Verilmezse,
        FeedbackRepository varsayılan DSN (docker-compose.yml'deki
        living-weather-postgres, port 5434) ile oluşturulur.
        """
        self.agent_name = "FL-Agent"
        self.repository = repository or FeedbackRepository()

    def record_prediction(self, reading: WeatherReading) -> FeedbackRecord:
        """
        Yeni bir tahmin geldiğinde, gelecekte değerlendirilmek üzere
        PostgreSQL'e kaydeder. Dönen FeedbackRecord'un db_id alanı
        artık PostgreSQL'deki gerçek satır id'sini taşıyor - bu id,
        evaluate_against_actual() çağrılırken hangi satırın
        güncelleneceğini belirlemek için kullanılıyor.
        """
        db_id = self.repository.insert_prediction(
            location=reading.location,
            latitude=reading.latitude,
            longitude=reading.longitude,
            predicted_temperature_c=reading.temperature_c,
            predicted_condition=reading.condition,
            source=reading.source,
            source_status=reading.source_status.value,
            confidence_score=reading.confidence_score,
            predicted_at=reading.timestamp,
        )
        return FeedbackRecord(db_id=db_id, prediction=reading)

    def evaluate_against_actual(
        self, record: FeedbackRecord, actual_temperature_c: float, actual_condition: str = ""
    ) -> AgentResponse:
        """
        Bir tahmini, sonradan öğrenilen gerçek değerle karşılaştırır.
        Kaynağa özel tolerans bandı uygular. Sonucu PostgreSQL'deki
        ilgili satıra (record.db_id) UPDATE eder.

        NOT: record.db_id None ise (örn. eski/bellek-içi bir kayıt,
        veya repository olmadan oluşturulmuş bir FeedbackRecord),
        veritabanı güncellemesi atlanır ama hesaplama yine de yapılır
        ve dönüş değeri (AgentResponse) her zaman doğru kalır - yani
        çağıran kod için davranış DEĞİŞMEZ, sadece kalıcılık opsiyonel.
        """
        source = record.prediction.source
        tolerance = TOLERANCE_BY_SOURCE.get(source, DEFAULT_TOLERANCE)

        eval_result = evaluate_numeric_tolerance(
            predicted=record.prediction.temperature_c,
            actual=actual_temperature_c,
            tolerance=tolerance,
            label=f"sıcaklık ({source})",
        )

        record.actual_temperature_c = actual_temperature_c
        record.actual_condition = actual_condition
        record.evaluated = True
        record.eval_score = eval_result.score
        record.eval_passed = eval_result.passed
        record.eval_reason = eval_result.reason

        if record.db_id is not None:
            self.repository.update_evaluation(
                record_id=record.db_id,
                actual_temperature_c=actual_temperature_c,
                actual_condition=actual_condition,
                eval_score=eval_result.score,
                eval_passed=eval_result.passed,
                eval_reason=eval_result.reason,
            )

        return AgentResponse(
            agent_name=self.agent_name,
            success=True,
            payload={
                "source": source,
                "passed": eval_result.passed,
                "score": eval_result.score,
                "deviation": eval_result.raw_deviation,
                "tolerance_used": tolerance,
            },
            confidence_score=eval_result.score / 5.0,
        )

    def calibration_report(self) -> dict:
        """
        Kaynağa göre gruplanmış kalibrasyon raporu. Hesaplama artık
        Python tarafında (eski self._history listesi üzerinde) DEĞİL,
        doğrudan PostgreSQL'de (SQL GROUP BY ile, bkz. db.py
        get_calibration_report) yapılıyor - kayıt sayısı büyüdükçe
        bu çok daha verimli, ve birden fazla worker/process aynı
        veritabanını okuduğu için rapor her zaman GÜNCEL ve PAYLAŞILAN
        bir görünüm sunuyor (eski bellek-içi versiyonda her worker
        kendi history'sini görürdü, raporlar TUTARSIZ olabilirdi).
        """
        return self.repository.get_calibration_report()


if __name__ == "__main__":
    from app.core.models import DataSourceStatus

    agent = FLAgent()

    print("=== TEST 1: Open-Meteo tahmini, küçük sapma (PASS bekleniyor) ===")
    reading1 = WeatherReading(
        location="İzmir", latitude=38.42, longitude=27.14,
        temperature_c=24.0, source="open-meteo",
        source_status=DataSourceStatus.HEALTHY, confidence_score=1.0,
    )
    record1 = agent.record_prediction(reading1)
    result1 = agent.evaluate_against_actual(record1, actual_temperature_c=25.5)
    print(f"Sonuç: {result1.payload}")

    print("\n=== TEST 2: Open-Meteo tahmini, büyük sapma (FAIL bekleniyor, sıkı tolerans) ===")
    reading2 = WeatherReading(
        location="İzmir", latitude=38.42, longitude=27.14,
        temperature_c=15.0, source="open-meteo",
        source_status=DataSourceStatus.HEALTHY, confidence_score=1.0,
    )
    record2 = agent.record_prediction(reading2)
    result2 = agent.evaluate_against_actual(record2, actual_temperature_c=25.0)
    print(f"Sonuç: {result2.payload}")

    print("\n=== TEST 3: LLM-Agent tahmini, AYNI büyük sapma (gevşek tolerans ile farklı skor) ===")
    reading3 = WeatherReading(
        location="İzmir", latitude=38.42, longitude=27.14,
        temperature_c=15.0, source="llm-agent-seasonal-estimate",
        source_status=DataSourceStatus.CRITICAL, confidence_score=0.2,
    )
    record3 = agent.record_prediction(reading3)
    result3 = agent.evaluate_against_actual(record3, actual_temperature_c=25.0)
    print(f"Sonuç: {result3.payload}")
    print(">>> Dikkat: Test 2 ve Test 3 AYNI sapmayı (10 derece) ölçüyor.")
    print(">>> Test 2 (open-meteo, tolerans=2.0): score=0.0  -- çok sıkı, sapma toleransı 5 katına çıkmış")
    print(">>> Test 3 (llm-agent, tolerans=8.0): score=1.88  -- daha gevşek tolerans, aynı sapma daha az 'cezalandırılıyor'")
    print(">>> İkisi de PASS eşiğini (tolerans aşıldığı için) geçemedi, ama SKOR farkı")
    print(">>> kaynağın doğasına göre adil bir karşılaştırma sağlıyor (LLM tahmininden zaten")
    print(">>> yüksek isabet beklenmiyor, bu yüzden aynı hata payı görece daha az ağır cezalandırılıyor).")

    print("\n=== TEST 4: Kalibrasyon raporu (kaynak bazlı özet) ===")
    report = agent.calibration_report()
    for source, stats in report.items():
        print(f"  {source}: pass_rate={stats['pass_rate']:.0%}, "
              f"avg_score={stats['average_score']:.2f}, n={stats['sample_size']}")
