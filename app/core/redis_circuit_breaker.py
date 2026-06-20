"""
app/core/redis_circuit_breaker.py
====================================
Living Weather - Redis Tabanlı Devre Kesici (Circuit Breaker)

NEDEN BU DOSYA AYRI, agent_di_agent.py İÇİNDEKİ CircuitBreaker
SİLİNMEDİ?
  agent_di_agent.py'deki CircuitBreaker sınıfı bellek-içi (in-memory)
  çalışıyordu ve zaten test edilmiş, kanıtlanmış bir mantığa sahipti
  (failure_threshold, cooldown_seconds, 3 durumlu state machine).
  O sınıfı SİLMEK yerine, AYNI PUBLIC ARAYÜZE (record_success,
  record_failure, can_attempt, .state) sahip YENİ bir sınıf yazıyoruz.
  Böylece DIAgent sınıfında SADECE HANGİ CircuitBreaker'ın
  kullanılacağı değişiyor (tek satır), state machine mantığının
  kendisi (ne zaman OPEN'a geçer, ne zaman HALF_OPEN'a döner) AYNI
  kalıyor - yani test edilmiş davranış bozulmuyor.

NEDEN REDIS GEREKLİ? (STATUS.md'deki bilinen risk)
  "DI-Agent'ın devre kesici durumu (circuit breaker state) şu an
  BELLEK-İÇİ - eğer FastAPI birden fazla worker/process ile çalışırsa
  her worker kendi devre durumunu tutar (paylaşılmaz)."

  Örnek senaryo: Cloud Run'da 4 worker process çalışıyor. Open-Meteo
  API'si çöktü. Worker-1, 3 art arda hata aldı, devresini OPEN yaptı.
  AMA Worker-2, 3, 4 bunu BİLMİYOR - onlar hâlâ kendi sayaçlarını
  sıfırdan tutuyor, hâlâ Open-Meteo'ya istek göndermeye çalışıyorlar
  (timeout bekleyerek). Devre kesicinin TÜM AMACI ("API çökmüşken
  sistemi yorma") bu durumda 4 worker'dan 3'ünde işe yaramıyor.

  Redis'e taşıyınca: TEK BİR merkezi state var, hangi worker isteği
  alırsa alsın, aynı Redis key'lerini okuyup yazıyor. Biri devreyi
  OPEN yaparsa, HEPSİ bunu anında görür.

NASIL ÇALIŞIYOR (Redis veri yapısı)?
  Her devre kesici örneği, kendine özel bir "key prefix" ile çalışır
  (örn. "circuit:di_agent:open-meteo"). Bu prefix altında 3 alan
  (Redis Hash - HSET/HGET) tutulur:
    - state          -> "closed" | "open" | "half_open"
    - failure_count   -> int (string olarak saklanır, Redis hep string tutar)
    - opened_at       -> float timestamp (epoch) veya boş

  NEDEN HASH (HSET) DEĞİL DE AYRI KEY'LER DE OLABİLİRDİ AMA HASH
  SEÇİLDİ: 3 alanı (state, failure_count, opened_at) ATOMİK olarak
  birlikte okuyup yazmak istiyoruz - bir worker "state oku" derken
  başka bir worker "failure_count yaz" yapmasın, yarış durumu
  (race condition) oluşmasın. Redis Hash, HGETALL ile tek seferde
  tüm alanları okumamızı sağlıyor.

FAIL-OPEN DAVRANIŞI (Redis'in KENDİSİ çökerse ne olur?):
  Eğer Redis'e bağlanılamıyorsa, devre kesici "CLOSED" (yani "normal,
  dene") davranışına FAIL-OPEN olur - yani Redis çökse bile DI-Agent
  Open-Meteo'yu denemeye çalışır (devre kesici hiç yokmuş gibi davranır).
  BU BİLİNÇLİ BİR KARARDIR: devre kesicinin amacı "performans
  optimizasyonu" (gereksiz denemeleri önlemek), GÜVENLİK KRİTİK bir
  mekanizma değil. Redis çöktüğünde DI-Agent'ın denemesi engellense
  (FAIL-CLOSED) sistem tamamen veri alamaz hale gelirdi - bu daha
  kötü bir sonuç olurdu. (Bu, policy_server.py'deki semantic check'in
  FAIL-CLOSED kararının TERSİ bir tercih - orada hassas veri/güvenlik
  riski olduğu için "reddet" tercih edilmişti; burada ise sadece bir
  performans optimizasyonu olduğu için "normal çalışmaya devam et"
  tercih ediliyor. İkisi de whitepaper'ın "Zero-Trust" felsefesinin
  farklı risk profillerine uygulanmış hali.)

Kaynak: Living Weather mimarisi + Day 5 whitepaper "Zero-Trust
        Development" felsefesinin çok-worker/dağıtık ortama taşınması.
"""

import time
from enum import Enum
from typing import Optional

import redis


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RedisCircuitBreaker:
    """
    agent_di_agent.py'deki CircuitBreaker ile AYNI PUBLIC ARAYÜZE
    sahip, ama durumu Redis'te (bellek-içi DEĞİL) tutan versiyon.

    Kullanım, eski CircuitBreaker ile birebir aynı:
        breaker = RedisCircuitBreaker(redis_client, key_prefix="circuit:di_agent")
        if breaker.can_attempt():
            ...
            breaker.record_success()  # veya breaker.record_failure()
    """

    def __init__(
        self,
        redis_client: "redis.Redis",
        key_prefix: str = "circuit:di_agent",
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
    ):
        self.redis = redis_client
        self.key = key_prefix  # Redis Hash key'i
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

    # ------------------------------------------------------------------
    # İÇ YARDIMCI: Redis'ten güvenli okuma (bağlantı hatasına dayanıklı)
    # ------------------------------------------------------------------
    def _read_state(self) -> dict:
        """
        Redis Hash'ten tüm alanları tek seferde okur.
        Redis'e erişilemezse (bağlantı hatası), FAIL-OPEN davranışı
        için "her şey normalmiş gibi" (CLOSED, sıfır hata) bir
        varsayılan döner - bkz. dosya başındaki FAIL-OPEN açıklaması.
        """
        try:
            raw = self.redis.hgetall(self.key)
        except redis.RedisError:
            return {"state": CircuitState.CLOSED.value, "failure_count": "0", "opened_at": ""}

        if not raw:
            # Key hiç yok demek - henüz hiç hata kaydedilmemiş, varsayılan CLOSED
            return {"state": CircuitState.CLOSED.value, "failure_count": "0", "opened_at": ""}

        # redis-py decode_responses=True ile çalıştığımızı varsayıyoruz
        # (yapılandırma get_redis_client() içinde yapılır), yani değerler
        # zaten str geliyor, bytes->str dönüşümüne gerek yok.
        return {
            "state": raw.get("state", CircuitState.CLOSED.value),
            "failure_count": raw.get("failure_count", "0"),
            "opened_at": raw.get("opened_at", ""),
        }

    def _write_state(self, state: CircuitState, failure_count: int, opened_at: Optional[float]) -> None:
        """
        Redis Hash'e tüm alanları tek seferde yazar (HSET ile mapping).
        Redis'e erişilemezse hatayı YUTAR (sessizce geç) - yazma
        başarısız olursa bile DI-Agent çalışmaya devam etmeli
        (FAIL-OPEN prensibi burada da geçerli: state kaybedilse bile
        sistem durmasın, en kötü ihtimalle devre kesici "unutkan"
        davranır, ama veri akışı kesilmez).
        """
        try:
            self.redis.hset(
                self.key,
                mapping={
                    "state": state.value,
                    "failure_count": str(failure_count),
                    "opened_at": "" if opened_at is None else str(opened_at),
                },
            )
        except redis.RedisError:
            pass  # FAIL-OPEN: Redis yazma hatası DI-Agent'ı durdurmamalı

    # ------------------------------------------------------------------
    # PUBLIC ARAYÜZ (eski CircuitBreaker ile birebir aynı imzalar)
    # ------------------------------------------------------------------
    @property
    def state(self) -> CircuitState:
        raw = self._read_state()
        return CircuitState(raw["state"])

    @property
    def failure_count(self) -> int:
        raw = self._read_state()
        try:
            return int(raw["failure_count"])
        except (ValueError, TypeError):
            return 0

    def record_success(self) -> None:
        self._write_state(CircuitState.CLOSED, failure_count=0, opened_at=None)

    def record_failure(self) -> None:
        raw = self._read_state()
        current_count = int(raw.get("failure_count") or 0)
        new_count = current_count + 1

        if new_count >= self.failure_threshold:
            self._write_state(CircuitState.OPEN, failure_count=new_count, opened_at=time.time())
        else:
            # Henüz eşiğe ulaşmadı, sayaç artıyor ama state CLOSED kalıyor
            self._write_state(CircuitState.CLOSED, failure_count=new_count, opened_at=None)

    def can_attempt(self) -> bool:
        """Şu an birincil kaynağı denemeye değer mi?"""
        raw = self._read_state()
        current_state = CircuitState(raw["state"])

        if current_state == CircuitState.CLOSED:
            return True

        if current_state == CircuitState.OPEN:
            opened_at_str = raw.get("opened_at") or ""
            try:
                opened_at = float(opened_at_str)
            except ValueError:
                # opened_at okunamıyorsa (bozuk veri), güvenli tarafta kal,
                # bir şans ver (HALF_OPEN'a benzer davranış)
                return True

            if time.time() - opened_at >= self.cooldown_seconds:
                # Cooldown bitti, HALF_OPEN'a geç (bir worker bunu yazar,
                # diğer worker'lar da bir sonraki okumada görür)
                self._write_state(CircuitState.HALF_OPEN, failure_count=self.failure_count, opened_at=opened_at)
                return True
            return False

        # HALF_OPEN: bir şans ver
        return True


def get_redis_client(
    host: str = "localhost",
    port: int = 6380,
    db: int = 0,
    socket_connect_timeout: float = 2.0,
) -> "redis.Redis":
    """
    Living Weather'ın Redis container'ına (docker-compose.yml'deki
    "living-weather-redis" servisi) bağlanan bir client döner.

    NOT: Port 6380 kullanıyoruz (6379 DEĞİL) - bilgisayarındaki başka
    projeler (örn. trucklife_redis) standart 6379 portunu kullanıyor,
    docker-compose.yml'de Living Weather'a özel 6380'e map ettik.

    decode_responses=True: Redis'ten gelen değerleri otomatik olarak
    bytes -> str çevirir, yukarıdaki kodda manuel decode() çağırmaya
    gerek kalmaz.
    """
    return redis.Redis(
        host=host,
        port=port,
        db=db,
        decode_responses=True,
        socket_connect_timeout=socket_connect_timeout,
    )


if __name__ == "__main__":
    print("=== TEST: RedisCircuitBreaker (gerçek Redis bağlantısı gerekir) ===")
    print("NOT: Bu testi çalıştırmadan önce 'docker compose up -d' ile")
    print("living-weather-redis container'ının ayakta olduğundan emin ol.\n")

    client = get_redis_client()
    try:
        client.ping()
        print("Redis bağlantısı OK.\n")
    except redis.RedisError as exc:
        print(f"UYARI: Redis'e bağlanılamadı ({exc}). FAIL-OPEN davranışı test edilecek.\n")

    # Temiz başlangıç için test key'ini sil
    client.delete("circuit:test")

    breaker = RedisCircuitBreaker(client, key_prefix="circuit:test", failure_threshold=2, cooldown_seconds=3.0)

    print(f"Başlangıç state: {breaker.state.value}, can_attempt: {breaker.can_attempt()}")

    print("\n-- 2 art arda hata kaydediliyor (threshold=2) --")
    breaker.record_failure()
    print(f"1. hata sonrası: state={breaker.state.value}, failure_count={breaker.failure_count}, can_attempt={breaker.can_attempt()}")
    breaker.record_failure()
    print(f"2. hata sonrası: state={breaker.state.value}, failure_count={breaker.failure_count}, can_attempt={breaker.can_attempt()}")
    print(">>> Beklenen: state=open, can_attempt=False (cooldown henüz bitmedi)")

    print("\n-- AYNI key_prefix ile YENİ bir breaker örneği oluşturuluyor --")
    print(">>> (Bu, farklı bir worker/process'i simüle ediyor - state PAYLAŞILMALI)")
    breaker2 = RedisCircuitBreaker(client, key_prefix="circuit:test", failure_threshold=2, cooldown_seconds=3.0)
    print(f"breaker2 (yeni örnek) state: {breaker2.state.value}, can_attempt: {breaker2.can_attempt()}")
    print(">>> Beklenen: state=open, can_attempt=False -- breaker1'in yazdığı durumu")
    print(">>> breaker2 GÖREBİLMELİ, çünkü ikisi de aynı Redis key'ini okuyor.")
    print(">>> (Bellek-içi versiyonda bu İMKANSIZ olurdu - her örnek kendi state'ini tutardı.)")

    print(f"\n-- {breaker.cooldown_seconds} saniye bekleniyor (cooldown) --")
    time.sleep(breaker.cooldown_seconds + 0.5)
    print(f"Cooldown sonrası can_attempt: {breaker.can_attempt()}, state: {breaker.state.value}")
    print(">>> Beklenen: can_attempt=True, state=half_open")

    breaker.record_success()
    print(f"\nBaşarı kaydedildikten sonra: state={breaker.state.value}, failure_count={breaker.failure_count}")
    print(">>> Beklenen: state=closed, failure_count=0")

    client.delete("circuit:test")
    print("\nTest key'i temizlendi.")
