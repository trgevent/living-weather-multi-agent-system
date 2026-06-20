"""
context_resolver.py
====================
Day 5 whitepaper pattern: "Context Hygiene & Prompt Sanitization"

NE İŞE YARAR?
-------------
Bir agent (LLM tabanlı) bir tool çağırırken, bazı argümanlar hassas
olabilir (email adresi, API key, telefon numarası vb). Bu değerleri
agent'ın prompt'una veya kodun içine hardcode etmek yerine, [[VARIABLE_NAME]]
şeklinde bir placeholder yazarız. Bu script, çalışma zamanında bu
placeholder'ları gerçek değerlerle (env değişkeni veya runtime state'ten)
güvenli bir şekilde değiştirir.

ÖRNEK KULLANIM SENARYOSU (Atlas / aile asistanı için):
  n8n workflow'unda Gmail bildirim şablonun şöyle olabilir:
    "Sayın [[PARENT_NAME]], [[STUDENT_NAME]] bugün [[EXAM_SCORE]] aldı."
  Bu şablonu LLM'e veya log'a yazarken placeholder olarak tutarsın,
  gerçek değerler sadece son anda enjekte edilir. Böylece LLM'in
  context'inde veya log dosyalarında çocuğunun gerçek adı/notu
  yanlışlıkla başka bir yere "sızmaz" (hallucination riski de düşer,
  çünkü LLM bu değerleri görmez, sadece placeholder'ı görür).

ÖRNEK KULLANIM SENARYOSU (Living Weather için):
  AS-Agent (Alert & Safety) bir uyarı mesajı oluştururken
  "[[USER_LOCATION]]" gibi bir placeholder kullanır, gerçek GPS
  koordinatı sadece gönderim anında enjekte edilir.

Kaynak: Google Day 5 whitepaper, "Implementing a Dynamic ContextResolver" (s.33-34)
"""

import os
import re
from typing import Optional, Dict, Any


def resolve_context(
    template_str: str,
    override_state: Optional[Dict[str, Any]] = None
) -> str:
    """
    [[VARIABLE_NAME]] şeklindeki placeholder'ları tarar ve değiştirir.

    Öncelik sırası:
      1. override_state sözlüğü (runtime'da elle verilen değerler)
      2. os.environ (ortam değişkenleri)
      3. Hiçbiri yoksa: placeholder'ı OLDUĞU GİBİ bırakır
         (sessizce boş string yapmak yerine - bu "silent failure"ı
         önler, eksik bir değişkeni hemen fark edersin)

    Args:
        template_str: İçinde [[VAR]] placeholder'ları olan metin
        override_state: Runtime'da öncelikli olacak değerler sözlüğü

    Returns:
        Placeholder'ları çözülmüş metin
    """
    if template_str is None:
        return ""

    state_to_check = override_state or {}

    def replacement(match: re.Match) -> str:
        var_name = match.group(1).strip()

        # 1. Önce runtime override'a bak
        if var_name in state_to_check and state_to_check[var_name] is not None:
            return str(state_to_check[var_name])

        # 2. Sonra ortam değişkenine bak
        elif var_name in os.environ and os.environ[var_name] is not None:
            return os.environ[var_name]

        # 3. Hiçbiri yoksa olduğu gibi bırak (silent failure önleme)
        else:
            return match.group(0)

    return re.sub(r"\[\[([^\]]+)\]\]", replacement, template_str)


def mask_pii(text: str, patterns: Optional[Dict[str, str]] = None) -> str:
    """
    BONUS fonksiyon (whitepaper'da yok ama Atlas için faydalı):
    Metindeki yaygın PII kalıplarını (email, telefon, TC no) otomatik
    olarak maskeler. Day 4'teki security_checkpoint node'undaki
    SSN/kredi kartı regex mantığının genel halidir.

    Args:
        text: Maskelenecek metin
        patterns: {isim: regex} sözlüğü, verilmezse varsayılanlar kullanılır

    Returns:
        PII'leri [MASKED_X] ile değiştirilmiş metin
    """
    default_patterns = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "TR_PHONE": r"(\+90|0)?\s?5\d{2}\s?\d{3}\s?\d{2}\s?\d{2}",
        "TC_NO": r"\b[1-9]\d{10}\b",  # 11 haneli TC kimlik no kalıbı
    }
    patterns = patterns or default_patterns

    masked = text
    for label, pattern in patterns.items():
        masked = re.sub(pattern, f"[MASKED_{label}]", masked)
    return masked


if __name__ == "__main__":
    # Hızlı manuel test - kanka bunu çalıştırınca ne göreceğini
    # aşağıda yorum olarak yazdım.

    print("=== TEST 1: Temel placeholder çözme ===")
    template = "Sayın [[PARENT_NAME]], öğrenci [[STUDENT_NAME]] sınavdan [[SCORE]] aldı."
    result = resolve_context(
        template,
        override_state={"PARENT_NAME": "Ahu Hanım", "STUDENT_NAME": "Ahmet", "SCORE": "85"}
    )
    print(result)
    # Beklenen çıktı:
    # Sayın Ahu Hanım, öğrenci Ahmet sınavdan 85 aldı.

    print("\n=== TEST 2: Eksik değişken (silent failure yok) ===")
    template2 = "API anahtarı: [[MISSING_KEY]]"
    result2 = resolve_context(template2, override_state={})
    print(result2)
    # Beklenen çıktı:
    # API anahtarı: [[MISSING_KEY]]   <-- değişmeden kaldı, hata yutulmadı

    print("\n=== TEST 3: Ortam değişkeninden okuma ===")
    os.environ["TEST_CITY"] = "İzmir"
    template3 = "Bugün hava durumu: [[TEST_CITY]]"
    result3 = resolve_context(template3)
    print(result3)
    # Beklenen çıktı:
    # Bugün hava durumu: İzmir

    print("\n=== TEST 4: PII maskeleme (bonus fonksiyon) ===")
    raw_text = "İletişim: deepbenchai@gmail.com, telefon 0532 123 45 67"
    masked = mask_pii(raw_text)
    print(masked)
    # Beklenen çıktı:
    # İletişim: [MASKED_EMAIL], telefon [MASKED_TR_PHONE]
