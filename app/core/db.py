"""
app/core/db.py
================
Living Weather - PostgreSQL Erişim Katmanı (Ham SQL + psycopg)

NEDEN ORM DEĞİL, HAM SQL?
  FL-Agent'ın ihtiyacı basit: tek bir tablo (feedback_records),
  birkaç INSERT/UPDATE/SELECT sorgusu. SQLAlchemy gibi bir ORM,
  bu ölçekte fayda sağlamadan ekstra bir öğrenme/bağımlılık katmanı
  eklerdi. Ham SQL + psycopg ile: sorgular AÇIKÇA görülebiliyor
  (ne çalıştığını gizlemiyor), bağımlılık sayısı az, debug etmek
  kolay (DBeaver'da aynı sorguları kopyala-yapıştır çalıştırabilirsin).

NE İŞE YARAR?
  - get_db_connection(): PostgreSQL'e (docker-compose.yml'deki
    "living-weather-postgres" servisi) bağlanan bir psycopg
    connection döner.
  - init_db(): schema.sql dosyasını çalıştırır (tablo yoksa oluşturur).
    Uygulama başlarken (FastAPI startup) bir kere çağrılır.
  - FeedbackRepository: FL-Agent'ın kullanacağı CRUD işlemleri
    (insert_prediction, update_evaluation, get_all_by_source, vb).

NEDEN "REPOSITORY" SINIFI (FLAgent'ın KENDİSİ DEĞİL)?
  agent_fl_agent.py'deki FLAgent sınıfı, "ajan mantığı" (tolerans
  bandı seçimi, evaluation_engine çağrısı) ile ilgilenmeli - VERİNİN
  NEREDE SAKLANDIĞI (bellek-içi liste mi, PostgreSQL mi) ile
  ilgilenmemeli. Bu ayrım (separation of concerns), ileride
  "PostgreSQL'den başka bir veritabanına geçersek" diye bir ihtiyaç
  çıkarsa, sadece bu dosyanın değişmesini sağlar - agent_fl_agent.py
  neredeyse hiç değişmez (zaten agent_master.py'deki tasarım
  felsefesiyle aynı: her bileşen TEK bir sorumluluğa sahip olmalı).

Kaynak: Living Weather mimarisi + Day 5 whitepaper'ın "single
        responsibility" / "her ajan tek işe odaklanır" prensibinin
        veri katmanına uygulanması.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row


# ----------------------------------------------------------------------
# BAĞLANTI YAPILANDIRMASI
# ----------------------------------------------------------------------
# NOT: Port 5434 kullanıyoruz (5432/5433 DEĞİL) - bilgisayarındaki
# başka projeler (medtrade-db, trucklife_db) standart portları
# kullanıyor, docker-compose.yml'de Living Weather'a özel 5434'e
# map ettik.
DEFAULT_DSN = (
    "host=localhost port=5434 dbname=living_weather "
    "user=living_weather password=living_weather_dev_pw"
)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_db_connection(dsn: str = DEFAULT_DSN) -> psycopg.Connection:
    """
    PostgreSQL'e yeni bir bağlantı açar.
    row_factory=dict_row: sorgu sonuçları tuple yerine dict olarak
    döner (örn. row["temperature_c"] yerine row[2] yazmak zorunda
    kalmayız - daha okunabilir, indeks hatalarına daha dayanıklı).
    """
    return psycopg.connect(dsn, row_factory=dict_row)


@contextmanager
def db_cursor(dsn: str = DEFAULT_DSN):
    """
    'with db_cursor() as cur:' şeklinde kullanılan yardımcı context
    manager. Bağlantıyı açar, cursor verir, iş bitince commit edip
    bağlantıyı kapatır. Hata olursa rollback yapar - yarım kalan bir
    INSERT/UPDATE veritabanında tutarsız veri bırakmasın diye.
    """
    conn = get_db_connection(dsn)
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(dsn: str = DEFAULT_DSN) -> None:
    """
    schema.sql dosyasını çalıştırır (CREATE TABLE IF NOT EXISTS,
    yani tablo zaten varsa hiçbir şey yapmaz - güvenle tekrar tekrar
    çağrılabilir). FastAPI startup'ında bir kere çağrılır.
    """
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with db_cursor(dsn) as cur:
        cur.execute(schema_sql)


# ----------------------------------------------------------------------
# FEEDBACK REPOSITORY (FL-Agent'ın kullanacağı CRUD işlemleri)
# ----------------------------------------------------------------------
class FeedbackRepository:
    """
    feedback_records tablosu üzerinde çalışan, ham SQL tabanlı
    erişim katmanı. FLAgent sınıfı bu repository'yi kullanır,
    SQL'in kendisiyle hiç ilgilenmez.
    """

    def __init__(self, dsn: str = DEFAULT_DSN):
        self.dsn = dsn

    def insert_prediction(
        self,
        location: str,
        latitude: float,
        longitude: float,
        predicted_temperature_c: float,
        predicted_condition: str,
        source: str,
        source_status: str,
        confidence_score: float,
        predicted_at: datetime,
    ) -> int:
        """
        Yeni bir tahmin kaydı ekler (henüz değerlendirilmemiş halde).
        Döndürdüğü id, sonradan evaluate_against_actual() çağrılırken
        kullanılır.
        """
        with db_cursor(self.dsn) as cur:
            cur.execute(
                """
                INSERT INTO feedback_records (
                    location, latitude, longitude,
                    predicted_temperature_c, predicted_condition,
                    source, source_status, confidence_score, predicted_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    location, latitude, longitude,
                    predicted_temperature_c, predicted_condition,
                    source, source_status, confidence_score, predicted_at,
                ),
            )
            row = cur.fetchone()
            return row["id"]

    def update_evaluation(
        self,
        record_id: int,
        actual_temperature_c: float,
        actual_condition: str,
        eval_score: float,
        eval_passed: bool,
        eval_reason: str,
    ) -> None:
        """
        Gerçek değer öğrenildiğinde, kaydı değerlendirme sonuçlarıyla
        günceller (evaluated=True işaretler).
        """
        with db_cursor(self.dsn) as cur:
            cur.execute(
                """
                UPDATE feedback_records
                SET actual_temperature_c = %s,
                    actual_condition = %s,
                    evaluated = TRUE,
                    eval_score = %s,
                    eval_passed = %s,
                    eval_reason = %s,
                    evaluated_at = now()
                WHERE id = %s
                """,
                (actual_temperature_c, actual_condition, eval_score, eval_passed, eval_reason, record_id),
            )

    def get_calibration_report(self) -> dict[str, dict[str, Any]]:
        """
        Kaynağa göre gruplanmış kalibrasyon istatistiklerini SQL
        tarafında (GROUP BY ile) hesaplar - tüm satırları Python'a
        çekip orada hesaplamak yerine, veritabanının kendisine
        hesaplattırmak, kayıt sayısı büyüdükçe çok daha verimli olur.
        """
        with db_cursor(self.dsn) as cur:
            cur.execute(
                """
                SELECT
                    source,
                    AVG(eval_score)                                    AS average_score,
                    AVG(CASE WHEN eval_passed THEN 1.0 ELSE 0.0 END)   AS pass_rate,
                    COUNT(*)                                           AS sample_size
                FROM feedback_records
                WHERE evaluated = TRUE
                GROUP BY source
                """
            )
            rows = cur.fetchall()

        return {
            row["source"]: {
                "pass_rate": float(row["pass_rate"] or 0.0),
                "average_score": float(row["average_score"] or 0.0),
                "sample_size": row["sample_size"],
            }
            for row in rows
        }

    def get_unevaluated(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Henüz gerçek değerle karşılaştırılmamış kayıtları döner.
        İleride "bekleyen değerlendirmeler" gibi bir endpoint/iş
        eklenirse kullanılabilir.
        """
        with db_cursor(self.dsn) as cur:
            cur.execute(
                """
                SELECT * FROM feedback_records
                WHERE evaluated = FALSE
                ORDER BY predicted_at ASC
                LIMIT %s
                """,
                (limit,),
            )
            return cur.fetchall()


if __name__ == "__main__":
    print("=== TEST: PostgreSQL erişim katmanı (gerçek DB bağlantısı gerekir) ===")
    print("NOT: Bu testi çalıştırmadan önce 'docker compose up -d' ile")
    print("living-weather-postgres container'ının ayakta olduğundan emin ol.\n")

    print("-- init_db() çağrılıyor (tablo yoksa oluşturulur) --")
    init_db()
    print("Tablo hazır.\n")

    repo = FeedbackRepository()

    print("=== TEST 1: Open-Meteo tahmini ekleniyor ===")
    id1 = repo.insert_prediction(
        location="İzmir", latitude=38.42, longitude=27.14,
        predicted_temperature_c=24.0, predicted_condition="clear",
        source="open-meteo", source_status="healthy", confidence_score=1.0,
        predicted_at=datetime.utcnow(),
    )
    print(f"Eklendi, id={id1}")

    print("\n=== TEST 2: Bu tahmin değerlendiriliyor (gerçek değer: 25.5°C) ===")
    repo.update_evaluation(
        record_id=id1, actual_temperature_c=25.5, actual_condition="clear",
        eval_score=4.5, eval_passed=True, eval_reason="sapma tolerans içinde",
    )
    print("Güncellendi.")

    print("\n=== TEST 3: LLM-Agent tahmini ekleniyor ve değerlendiriliyor ===")
    id2 = repo.insert_prediction(
        location="İzmir", latitude=38.42, longitude=27.14,
        predicted_temperature_c=15.0, predicted_condition="unknown",
        source="llm-agent-seasonal-estimate", source_status="critical", confidence_score=0.2,
        predicted_at=datetime.utcnow(),
    )
    repo.update_evaluation(
        record_id=id2, actual_temperature_c=25.0, actual_condition="clear",
        eval_score=1.88, eval_passed=False, eval_reason="tolerans aşıldı ama gevşek tolerans",
    )
    print(f"Eklendi ve güncellendi, id={id2}")

    print("\n=== TEST 4: Kalibrasyon raporu (SQL GROUP BY ile, kaynağa göre) ===")
    report = repo.get_calibration_report()
    for source, stats in report.items():
        print(f"  {source}: pass_rate={stats['pass_rate']:.0%}, "
              f"avg_score={stats['average_score']:.2f}, n={stats['sample_size']}")
    print("\n>>> Bu rapor artık PostgreSQL'den geliyor - uygulama yeniden")
    print(">>> başlasa bile bu veri KAYBOLMAYACAK (bellek-içi versiyondan farkı budur).")
    print(">>> DBeaver'dan 'SELECT * FROM feedback_records;' ile aynı veriyi görebilirsin.")
