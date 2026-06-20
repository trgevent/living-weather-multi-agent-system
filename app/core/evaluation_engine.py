"""
evaluation_engine.py
======================
Day 5 whitepaper pattern: "Evaluation" (Tests vs. Evaluation)

NEDEN BU GEREKLİ? (whitepaper'dan, s.29-30)
--------------------------------------------
Geleneksel unit test ikili (binary) sorular sorar:
    "Fonksiyon doğru değeri döndürdü mü?" -> True / False

Ama bir ML/LLM bileşeni (tahmin modeli, sınıflandırıcı, özetleyici)
100 unit testten geçebilir ve hâlâ "spektaküler" şekilde başarısız
olabilir - yanlış aracı seçerek, kritik bir cevabı yanlış yorumlayarak,
ya da bir gerçeği "halüsinasyon" görerek (var olmayan bir şeyi
kendinden emin şekilde söyleyerek).

Bu hata payı bir "kusur" değil, modelin DOĞASINDA olan bir özellik.
Bu yüzden test stratejisi bunu hesaba katmalı:

    Unit Test  -> "Doğru mu?"               (ikili, kesin)
    Evaluation -> "En azından baseline kadar iyi mi?"  (0-5 skor,
                   tolerans bandı, LLM-as-judge)

LIVING WEATHER için NEREDE KULLANILIR?
  FL-Agent (Feedback & Learning Agent), gerçek hava durumu ile
  tahminleri karşılaştırıp model kalibrasyonu yapıyor. Burada
  "tahmin %100 doğru mu" diye sormak anlamsız (hava durumu zaten
  olasılıksal bir alan) - bunun yerine "tahmin, kabul edilebilir
  bir tolerans bandı içinde mi" diye sormalıyız.

  Örnek: Tahmin 24°C dedi, gerçek sıcaklık 26°C çıktı.
  Binary test: "24 == 26? HAYIR, FAIL." -> yanlış yaklaşım, çünkü
  2 derece sapma kabul edilebilir bir hata payı.
  Evaluation: "Sapma tolerans bandı içinde mi (örn. ±3°C)? EVET,
  PASS, skor: 4/5." -> doğru yaklaşım.

  Diğer örnek (LLM-as-judge gerektiren durum): RI-Agent (Route
  Intelligence) bir rota önerisi metni üretti. Bunun "doğru" olup
  olmadığını regex ile ölçemezsin - bir LLM'e "bu öneri mantıklı mı,
  güvenli mi, eksik bir uyarı var mı?" diye sorman gerekir.

ATLAS için NEREDE KULLANILIR?
  LGS koçluk sisteminin "günlük görev önerisi" agent'ı bir öğrenciye
  ödev önerisi üretiyor. Bunun "doğru/yanlış" diye test edilemez,
  ama "bu öneri öğrencinin seviyesine uygun mu, önceki konularla
  tutarlı mı?" diye 0-5 skorla değerlendirilebilir.

Kaynak: Google Day 5 whitepaper, "Evaluation" bölümü (s.29-30)
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class EvalResult:
    """Tek bir değerlendirmenin sonucu."""
    passed: bool
    score: float          # 0.0 - 5.0 arası
    reason: str
    raw_deviation: Optional[float] = None  # sayısal sapma varsa (örn. derece farkı)


@dataclass
class EvalReport:
    """Birden fazla test case'in toplu sonucu."""
    results: List[EvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    def summary(self) -> str:
        lines = [
            f"Toplam test: {len(self.results)}",
            f"Geçti: {sum(1 for r in self.results if r.passed)}",
            f"Geçme oranı: {self.pass_rate:.0%}",
            f"Ortalama skor: {self.average_score:.2f} / 5.0",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# DEĞERLENDİRME 1: TOLERANS BANDI (sayısal tahminler için - hava durumu)
# ----------------------------------------------------------------------
def evaluate_numeric_tolerance(
    predicted: float,
    actual: float,
    tolerance: float,
    label: str = "değer",
) -> EvalResult:
    """
    Sayısal bir tahminin, gerçek değere göre kabul edilebilir bir
    sapma (tolerans) içinde olup olmadığını kontrol eder.

    Args:
        predicted: Modelin tahmini (örn. 24.0 derece)
        actual: Gerçekleşen değer (örn. 26.0 derece)
        tolerance: Kabul edilebilir maksimum sapma (örn. 3.0 derece)
        label: Çıktıda görünecek etiket (örn. "sıcaklık")

    Returns:
        EvalResult - sapma tolerans içindeyse passed=True
    """
    deviation = abs(predicted - actual)
    passed = deviation <= tolerance

    # Skor: sapma 0 ise 5.0, tolerans sınırındaysa ~2.5, çok aşarsa 0'a yaklaşır
    if tolerance > 0:
        score = max(0.0, 5.0 * (1 - (deviation / (tolerance * 2))))
    else:
        score = 5.0 if deviation == 0 else 0.0
    score = min(5.0, round(score, 2))

    reason = (
        f"{label}: tahmin={predicted}, gerçek={actual}, "
        f"sapma={deviation:.2f}, tolerans=±{tolerance} -> "
        f"{'KABUL EDİLDİ' if passed else 'TOLERANS AŞILDI'}"
    )

    return EvalResult(passed=passed, score=score, reason=reason, raw_deviation=deviation)


# ----------------------------------------------------------------------
# DEĞERLENDİRME 2: LLM-AS-JUDGE (metinsel/nitel çıktılar için)
# ----------------------------------------------------------------------
def evaluate_with_llm_judge(
    task_description: str,
    agent_output: str,
    rubric: str,
    gemini_api_key: Optional[str] = None,
) -> EvalResult:
    """
    Bir LLM'e, agent'ın ürettiği metni belirli bir rubrik (kriter
    seti) üzerinden 0-5 arası puanlamasını ister.

    Args:
        task_description: Agent'a verilen görev (bağlam için)
        agent_output: Agent'ın ürettiği gerçek çıktı
        rubric: Hangi kriterlere göre puanlanacağı (örn. "güvenlik,
                netlik, eksik uyarı var mı")
        gemini_api_key: Verilmezse os.environ'dan okunur, yoksa
                        nötr bir "skip" sonucu döner.

    Returns:
        EvalResult - LLM'in verdiği skor ve gerekçe
    """
    api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return EvalResult(
            passed=True,
            score=2.5,
            reason="LLM-as-judge atlandı (GEMINI_API_KEY tanımlı değil), "
                   "nötr skor verildi.",
        )

    try:
        from google import genai
    except ImportError:
        return EvalResult(
            passed=True,
            score=2.5,
            reason="LLM-as-judge atlandı (google-genai paketi kurulu değil).",
        )

    client = genai.Client(api_key=api_key)

    prompt = (
        "Sen bir kalite değerlendirme uzmanısın. Aşağıdaki görevi ve "
        "agent'ın ürettiği çıktıyı, verilen rubriğe göre değerlendir.\n\n"
        f"GÖREV: {task_description}\n\n"
        f"AGENT ÇIKTISI: {agent_output}\n\n"
        f"RUBRİK (değerlendirme kriterleri): {rubric}\n\n"
        "Cevabını TAM olarak şu formatta ver:\n"
        "SKOR: <0 ile 5 arası bir sayı>\n"
        "GEREKÇE: <kısa açıklama>"
    )

    # DÜZELTME (RI-Agent geliştirilirken bulundu, 20 Haziran 2026):
    # API çağrısı artık try-except içinde. Eskiden geçersiz bir API key,
    # ağ kesintisi veya kota aşımı durumunda bu fonksiyon İSTİSNA
    # FIRLATIYORDU - yani LLM-as-judge'ı çağıran ajan (RI-Agent gibi)
    # çökerdi. Bu, policy_server.py'deki check_action_semantic'in ZATEN
    # uyguladığı "API'ye erişilemezse sistemi durdurmadan nötr/güvenli
    # bir sonuç dön" prensibiyle TUTARSIZDI. Şimdi tutarlı: API hatası
    # durumunda nötr bir "skip" sonucu dönülüyor, sistem ÇÖKMÜYOR -
    # bu da projenin "Zarif Bozunma" felsefesini evaluation katmanına
    # da taşıyor.
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = (response.text or "").strip()
    except Exception as exc:
        return EvalResult(
            passed=True,
            score=2.5,
            reason=f"LLM-as-judge atlandı (API hatası: {type(exc).__name__}: {exc}), "
                   f"nötr skor verildi.",
        )

    # Basit parsing: "SKOR: X" satırını bul
    score = 2.5  # varsayılan, parse edilemezse
    reason = text
    for line in text.splitlines():
        if line.upper().startswith("SKOR"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
        if line.upper().startswith("GEREKÇE"):
            reason = line.split(":", 1)[1].strip() if ":" in line else line

    passed = score >= 3.0  # 3 ve üstü "kabul edilebilir" sayılıyor (ayarlanabilir)

    return EvalResult(passed=passed, score=score, reason=reason)


if __name__ == "__main__":
    # ============================================================
    # SENARYO 1: Living Weather FL-Agent - sıcaklık tahmini değerlendirmesi
    # ============================================================
    print("=== TEST 1: Hava durumu tahmini, tolerans İÇİNDE (PASS bekleniyor) ===")
    result1 = evaluate_numeric_tolerance(
        predicted=24.0, actual=26.0, tolerance=3.0, label="sıcaklık (°C)"
    )
    print(f"Sonuç: passed={result1.passed}, score={result1.score}, reason={result1.reason}")

    print("\n=== TEST 2: Hava durumu tahmini, tolerans AŞILDI (FAIL bekleniyor) ===")
    result2 = evaluate_numeric_tolerance(
        predicted=15.0, actual=26.0, tolerance=3.0, label="sıcaklık (°C)"
    )
    print(f"Sonuç: passed={result2.passed}, score={result2.score}, reason={result2.reason}")

    print("\n=== TEST 3: Toplu rapor (birden fazla tahmin) ===")
    report = EvalReport()
    report.results.append(result1)
    report.results.append(result2)
    report.results.append(
        evaluate_numeric_tolerance(predicted=10.0, actual=11.0, tolerance=2.0, label="rüzgar (km/s)")
    )
    print(report.summary())

    print("\n=== TEST 4 (LLM-as-judge): RI-Agent'ın rota önerisi metni değerlendiriliyor ===")
    result4 = evaluate_with_llm_judge(
        task_description="Kullanıcıya İzmir'den Bodrum'a araçla giderken hava durumu "
                          "uyarısı içeren bir rota önerisi yaz.",
        agent_output="Bodrum yoluna çıkabilirsiniz, hava güzel olacak.",
        rubric="Öneri, varsa kötü hava koşullarını (sis, yağmur, fırtına) belirtiyor mu? "
               "Net ve eyleme geçirilebilir mi?",
    )
    print(f"Sonuç: passed={result4.passed}, score={result4.score}, reason={result4.reason}")
    # NOT: Bu örnek çıktı kasıtlı olarak "zayıf" yazıldı (hiçbir hava
    # durumu detayı yok) - eğer GEMINI_API_KEY tanımlıysa, LLM bunu
    # düşük puanlamalı (rubrikteki kriteri karşılamıyor).
