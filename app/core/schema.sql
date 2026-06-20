-- app/core/schema.sql
-- =====================
-- Living Weather - PostgreSQL Şeması
--
-- NEDEN BU TABLO?
--   FL-Agent (Feedback & Learning Agent), şu ana kadar tahmin
--   geçmişini (FeedbackRecord) sadece bellek-içi bir Python listesinde
--   tutuyordu (app/agents/fl_agent/agent.py içindeki self._history).
--   Uygulama yeniden başladığında bu liste SİLİNİYORDU - yani
--   kalibrasyon raporu ("kaynak X'in ortalama skoru Y") hiçbir zaman
--   birden fazla oturum boyunca biriken gerçek veriye dayanamıyordu.
--
--   Bu tablo, core/models.py'deki FeedbackRecord Pydantic modelinin
--   BİREBİR karşılığıdır - aynı alanlar, aynı anlam.
--
-- ÇALIŞTIRMA:
--   Bu dosya, app/core/db.py içindeki init_db() fonksiyonu tarafından
--   otomatik çalıştırılır (uygulama başlarken tablo yoksa oluşturulur).
--   Elle çalıştırmak istersen (örn. DBeaver üzerinden):
--     1. DBeaver'da yeni bir PostgreSQL bağlantısı aç:
--        Host: localhost, Port: 5434, Database: living_weather
--        User: living_weather, Password: living_weather_dev_pw
--     2. Bu dosyanın içeriğini SQL Editor'e yapıştırıp çalıştır.

CREATE TABLE IF NOT EXISTS feedback_records (
    id                      SERIAL PRIMARY KEY,

    -- WeatherReading (tahmin) alanları - core/models.py WeatherReading ile eşleşir
    location                TEXT NOT NULL,
    latitude                DOUBLE PRECISION NOT NULL,
    longitude               DOUBLE PRECISION NOT NULL,
    predicted_temperature_c DOUBLE PRECISION NOT NULL,
    predicted_condition     TEXT NOT NULL DEFAULT 'unknown',
    source                  TEXT NOT NULL,             -- örn. "open-meteo", "llm-agent-seasonal-estimate"
    source_status           TEXT NOT NULL,
    confidence_score        DOUBLE PRECISION NOT NULL,
    predicted_at            TIMESTAMPTZ NOT NULL,       -- WeatherReading.timestamp

    -- Değerlendirme (evaluation) alanları - sonradan doldurulur
    actual_temperature_c    DOUBLE PRECISION,
    actual_condition        TEXT,
    evaluated               BOOLEAN NOT NULL DEFAULT FALSE,
    eval_score              DOUBLE PRECISION,
    eval_passed             BOOLEAN,
    eval_reason             TEXT,
    evaluated_at            TIMESTAMPTZ,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Kalibrasyon raporu (calibration_report) sorgusu "source" alanına
-- göre GROUP BY yapacağı için bir indeks ekliyoruz - tablo büyüdükçe
-- bu sorguyu hızlandırır.
CREATE INDEX IF NOT EXISTS idx_feedback_records_source
    ON feedback_records (source);

-- Henüz değerlendirilmemiş kayıtları bulmak (evaluate_against_actual
-- çağrılacak kayıtları seçmek) için faydalı bir indeks.
CREATE INDEX IF NOT EXISTS idx_feedback_records_evaluated
    ON feedback_records (evaluated);
