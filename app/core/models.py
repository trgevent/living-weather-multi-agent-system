"""
app/core/models.py
====================
Living Weather - Paylaşılan Veri Modelleri

NE İŞE YARAR?
-------------
Her ajan birbirine veri gönderirken aynı "dil"i konuşmalı. Bu dosya,
tüm ajanların ortak kullandığı veri yapılarını (Pydantic modelleri)
tanımlar. Pydantic kullanıyoruz çünkü:
  1. Otomatik veri doğrulama (validation) yapar - yanlış tipte veri
     gelirse hemen hata verir, sessizce yanlış veriyle ilerlemez
  2. JSON serialize/deserialize otomatik (FastAPI + Redis ile uyumlu)
  3. ADK 2.0 ve diğer agent framework'leri de Pydantic'i destekliyor
     (Day 4'teki ambient-expense-agent'ta da görmüştük)

MİMARİ NOT:
  Her ajan, WeatherContext nesnesini alır, kendi alanını doldurur/
  günceller, Master Agent'a geri döner. Bu "paylaşılan kara tahta"
  (shared blackboard) deseni, multi-agent sistemlerde yaygın bir
  koordinasyon yöntemidir.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class DataSourceStatus(str, Enum):
    """
    DI-Agent'ın devre kesici (circuit breaker) durumunu izlemek için.
    Whitepaper'daki "Zarif Bozunma" felsefesinin temel taşı.
    """
    HEALTHY = "healthy"           # Birincil kaynak (Open-Meteo) çalışıyor
    DEGRADED = "degraded"         # Birincil çöktü, yedek kaynağa geçildi
    CRITICAL = "critical"         # Tüm dış kaynaklar çöktü, LLM-Agent devrede
    UNKNOWN = "unknown"           # Henüz hiç sorgu yapılmadı


class AlertSeverity(str, Enum):
    """AS-Agent'ın uyarı seviyeleri."""
    INFO = "info"
    WARNING = "warning"
    SEVERE = "severe"
    EXTREME = "extreme"


class WeatherReading(BaseModel):
    """
    Tek bir hava durumu ölçümü/tahmini.
    DI-Agent bunu üretir, diğer ajanlar tüketir.
    """
    location: str
    latitude: float
    longitude: float
    temperature_c: float
    feels_like_c: Optional[float] = None
    humidity_pct: Optional[float] = Field(default=None, ge=0, le=100)
    wind_speed_kmh: Optional[float] = None
    precipitation_mm: Optional[float] = None
    condition: str = "unknown"  # örn. "clear", "rain", "storm"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "unknown"      # örn. "open-meteo", "visualcrossing", "llm-fallback"
    source_status: DataSourceStatus = DataSourceStatus.UNKNOWN
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)


class AgentResponse(BaseModel):
    """
    Her ajanın Master Agent'a döndürdüğü standart zarf (envelope).
    Hangi ajan, ne zaman, ne üretti, ne kadar güvenilir - hepsi burada.
    """
    agent_name: str
    success: bool
    payload: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    error_message: Optional[str] = None
    processing_time_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WeatherAlert(BaseModel):
    """AS-Agent'ın ürettiği uyarı nesnesi."""
    severity: AlertSeverity
    title: str
    description: str
    location: str
    triggered_by: str  # hangi koşul/kural tetikledi
    requires_human_review: bool = False  # Policy Server kararı


class FeedbackRecord(BaseModel):
    """
    FL-Agent'ın değerlendirme/kalibrasyon kaydı.
    Tahmin vs gerçek karşılaştırması için.
    """
    # db_id: PostgreSQL'deki feedback_records.id alanına karşılık gelir.
    # Henüz veritabanına yazılmamış (örn. PostgreSQL'siz çalışan eski
    # bellek-içi mod) bir kayıt için None kalabilir - GERİYE DÖNÜK
    # UYUMLULUK için opsiyonel eklendi (varsayılan None, eski kodu bozmaz).
    db_id: Optional[int] = None
    prediction: WeatherReading
    actual_temperature_c: Optional[float] = None
    actual_condition: Optional[str] = None
    evaluated: bool = False
    eval_score: Optional[float] = None  # 0-5 arası, evaluation_engine.py'dan
    eval_passed: Optional[bool] = None
    eval_reason: Optional[str] = None


class WeatherContext(BaseModel):
    """
    "Paylaşılan kara tahta" - tüm ajanların okuyup yazdığı ana nesne.
    Master Agent, bu nesneyi her ajana sırayla (veya paralel) geçirir.
    """
    request_id: str
    location_query: str  # kullanıcının girdiği konum (örn. "İzmir")
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Ajanların doldurduğu alanlar
    current_reading: Optional[WeatherReading] = None
    forecast_readings: List[WeatherReading] = Field(default_factory=list)
    alerts: List[WeatherAlert] = Field(default_factory=list)
    feedback_records: List[FeedbackRecord] = Field(default_factory=list)

    # Her ajanın çalışma kaydı (debugging ve audit için)
    agent_trace: List[AgentResponse] = Field(default_factory=list)

    # Sistem durumu
    overall_status: DataSourceStatus = DataSourceStatus.UNKNOWN
    created_at: datetime = Field(default_factory=datetime.utcnow)
