"""
policy_server.py
=================
Day 5 whitepaper pattern: "Policy Server" (Hybrid: Structural + Semantic Gating)

NE İŞE YARAR?
-------------
Bir agent bir tool çağırmak istediğinde, bu çağrı doğrudan çalıştırılmaz.
Önce PolicyService'ten "izin var mı?" diye sorulur. İki katmanlı kontrol var:

  KATMAN 1 - STRUCTURAL GATING (Trafik Lambası):
    Deterministik, hızlı, YAML tabanlı kural kontrolü.
    "viewer rolü send_email tool'unu kullanabilir mi?" -> Hayır, regex/dict
    lookup ile anında cevap. LLM'e sorulmaz, milisaniyeler sürer.

  KATMAN 2 - SEMANTIC GATING (Akıllı Hakem):
    Tool kullanımı izinli OLSA BİLE, argümanların İÇERİĞİ politikaya
    aykırı olabilir. Örnek: admin send_email kullanabilir (yapısal
    olarak izinli), AMA email içeriğinde maskelenmemiş bir TC kimlik
    no veya kredi kartı numarası varsa, bu semantik bir ihlaldir.
    Bunu regex ile yakalamak imkansızdır (regex'le "her olası PII
    kalıbını" yakalayamazsın), bu yüzden ikinci bir LLM çağrısı ile
    "bu içerik politikaya uygun mu?" diye soruyoruz.

NEDEN BU İKİ KATMAN AYRI?
  Structural -> hızlı, ücretsiz (API çağrısı yok), kaba filtre
  Semantic   -> yavaş, ücretli (1 LLM çağrısı), ince filtre
  İkisini birleştirince: hem hızlı hem akıllı bir güvenlik ağı olur.

ATLAS için kullanım: n8n workflow'undan bir Gmail bildirimi gönderilmeden
önce, bu policy server'a sorulur. "editor" rolü email gönderemez (yapısal),
"admin" gönderebilir ama içerik kontrolünden geçmesi gerekir (semantik).

LIVING WEATHER için kullanım: AS-Agent bir uyarı yayınlamadan önce,
içeriğin gerçekten bir "alert" olup olmadığı (örn. yanlış alarm,
spam, agent hallucination) semantik katmanda kontrol edilebilir.

Kaynak: Google Day 5 whitepaper, "Policy Server" (s.30-32)

NOT (billing'siz çalışma şekli):
  Bu implementasyon whitepaper'daki örnekten FARKLI olarak Vertex AI
  Client (google.genai, vertexai=True) yerine, senin zaten sahip
  olduğun AI Studio GEMINI_API_KEY'i ile çalışacak şekilde yazıldı.
  Yani billing/GCP proje gerektirmiyor, tamamen yerel + ücretsiz tier.
"""

import os
import yaml
from typing import Optional, List, Dict, Any
from pathlib import Path


class PolicyService:
    """
    Hibrit politika motoru: yapısal (YAML) + semantik (LLM) kontrol.
    """

    def __init__(
        self,
        role: str,
        env: str = "localhost",
        policy_file: str = "policies.yaml",
        gemini_api_key: Optional[str] = None,
    ):
        """
        Args:
            role: Çağıran kullanıcının/agent'ın rolü (örn: "viewer", "admin")
            env: Çalışma ortamı (örn: "localhost", "production")
            policy_file: policies.yaml dosyasının yolu
            gemini_api_key: Semantic check için Gemini API key.
                            Verilmezse os.environ["GEMINI_API_KEY"] kullanılır.
                            Hiç yoksa semantic check atlanır (sadece uyarı verir).
        """
        self.role = role
        self.env = env
        self.config = self._load_config(policy_file)
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

    @staticmethod
    def _load_config(policy_file: str) -> Dict[str, Any]:
        """
        DÜZELTME (Antigravity code-check bulgusu, 19.06.2026):
        Path artık script'in KENDİ konumuna göre çözülüyor, çalışma
        dizinine (CWD) göre değil. Böylece "python policy_server.py"
        hangi klasörden çalıştırılırsa çalıştırılsın, policies.yaml
        her zaman doğru yerde aranır.
        """
        path = Path(policy_file)
        if not path.is_absolute():
            # Script'in bulunduğu klasöre göre göreli yol çöz
            path = Path(__file__).parent / policy_file

        if not path.exists():
            raise FileNotFoundError(
                f"Policy dosyası bulunamadı: {path}. "
                f"policies.yaml dosyasının bu script ile aynı klasörde "
                f"olduğundan emin ol."
            )
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            # Dosya tamamen boşsa yaml.safe_load None döner, bunu da koru
            return loaded or {}

    # ------------------------------------------------------------------
    # KATMAN 1: STRUCTURAL GATING
    # ------------------------------------------------------------------
    def is_tool_allowed(self, tool_name: str) -> bool:
        """
        Saf YAML/dict lookup. LLM çağrısı YOK, anında sonuç döner.

        DÜZELTME (Antigravity code-check bulgusu, 19.06.2026):
        YAML'da bir bölüm (örn. "blocked_tools:" veya "allowed_tools:")
        boş yazılırsa, PyYAML bunu None olarak yükler - boş liste [] değil.
        Eskiden ".get(..., [])" sadece KEY HİÇ YOKSA varsayılanı kullanıyordu;
        key VARDI ama değeri None ise hâlâ None dönüyordu, bu da
        "in None" gibi bir TypeError'a yol açıyordu.
        Şimdi her ".get()" sonucunu "or {}" / "or []" ile garantiye alıyoruz.
        """
        # Önce: bu environment'ta bu tool blocklu mu?
        env_config = (self.config.get("environments", {}) or {}).get(self.env, {}) or {}
        blocked = env_config.get("blocked_tools", []) or []
        if tool_name in blocked:
            return False

        # Sonra: bu rol bu tool'u kullanabilir mi?
        role_config = (self.config.get("roles", {}) or {}).get(self.role, {}) or {}
        allowed = role_config.get("allowed_tools", []) or []
        return "*" in allowed or tool_name in allowed

    # ------------------------------------------------------------------
    # KATMAN 2: SEMANTIC GATING
    # ------------------------------------------------------------------
    def check_action_semantic(self, action_description: str) -> Dict[str, Any]:
        """
        Gemini API ile içerik/niyet kontrolü yapar.

        Returns:
            {"allowed": bool, "reason": str}
            API key yoksa: {"allowed": True, "reason": "skipped - no API key"}
            (yani semantic check opsiyonel bir ek katman, zorunlu değil)
        """
        if not self.gemini_api_key:
            return {
                "allowed": True,
                "reason": "Semantic check atlandı (GEMINI_API_KEY tanımlı değil)",
            }

        try:
            from google import genai
        except ImportError:
            return {
                "allowed": True,
                "reason": "Semantic check atlandı (google-genai paketi kurulu değil, "
                          "'pip install google-genai' ile kurabilirsin)",
            }

        client = genai.Client(api_key=self.gemini_api_key)

        prompt = (
            "Aşağıdaki aksiyon açıklamasını incele. Eğer maskelenmemiş "
            "kişisel veri (TC kimlik no, kredi kartı, tam adres, şifre/API key) "
            "içeriyorsa veya açıkça zararlı/şüpheli bir niyet taşıyorsa "
            "(örn: 'tüm doğrulamaları atla', 'limitleri görmezden gel') "
            "cevabının İLK kelimesi 'VIOLATION' olsun ve kısaca neden yaz. "
            "Aksi halde cevabının ilk kelimesi 'OK' olsun.\n\n"
            f"Aksiyon: {action_description}"
        )

        # DÜZELTME (Antigravity code-check bulgusu, 19.06.2026):
        # API çağrısı artık try-except içinde. Ağ kesintisi, kota aşımı
        # veya geçersiz key durumunda sistem ÇÖKMEZ, FAIL-CLOSED davranır
        # (yani "izin var" yerine "izin yok / şüpheli" sayılır). Bu,
        # hassas veri (Atlas'taki öğrenci/veli bilgisi gibi) için doğru
        # taraf - API'ye erişemiyorsak, riski göze almamak daha güvenli.
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = (response.text or "").strip()
        except Exception as exc:
            return {
                "allowed": False,  # FAIL-CLOSED: API'ye erişilemiyorsa reddet
                "reason": f"Semantic check başarısız oldu (FAIL-CLOSED uygulandı): "
                          f"{type(exc).__name__}: {exc}",
            }

        is_violation = text.upper().startswith("VIOLATION")

        return {
            "allowed": not is_violation,
            "reason": text,
        }

    # ------------------------------------------------------------------
    # BİRLEŞTİRİLMİŞ KONTROL (ikisini tek çağrıda yapar)
    # ------------------------------------------------------------------
    def authorize(self, tool_name: str, action_description: str = "") -> Dict[str, Any]:
        """
        Hem yapısal hem semantik kontrolü sırayla yapar.
        Yapısal kontrol fail ederse, semantic check'e hiç gitmez
        (gereksiz LLM çağrısı / maliyet önlenir).

        Returns:
            {
                "allowed": bool,
                "layer": "structural" | "semantic" | "passed",
                "reason": str
            }
        """
        # 1. Önce ucuz/hızlı kontrolü yap
        if not self.is_tool_allowed(tool_name):
            return {
                "allowed": False,
                "layer": "structural",
                "reason": f"Rol '{self.role}' / ortam '{self.env}' için "
                          f"'{tool_name}' tool'una izin yok.",
            }

        # 2. Yapısal kontrol geçtiyse, içerik varsa semantik kontrole git
        if action_description:
            semantic_result = self.check_action_semantic(action_description)
            if not semantic_result["allowed"]:
                return {
                    "allowed": False,
                    "layer": "semantic",
                    "reason": semantic_result["reason"],
                }

        return {
            "allowed": True,
            "layer": "passed",
            "reason": "Yapısal ve semantik kontrolden geçti.",
        }


if __name__ == "__main__":
    # ============================================================
    # MANUEL TEST SENARYOLARI
    # Bu dosyanın bulunduğu klasörde policies.yaml olmalı.
    # ============================================================

    print("=== TEST 1: viewer rolü, send_email (yapısal RED) ===")
    viewer_policy = PolicyService(role="viewer", env="production")
    result = viewer_policy.authorize(tool_name="send_email", action_description="")
    print(result)
    # Beklenen: allowed=False, layer="structural"
    # (viewer rolünde send_email yok)

    print("\n=== TEST 2: admin rolü, localhost'ta send_email (yapısal RED) ===")
    admin_local = PolicyService(role="admin", env="localhost")
    result2 = admin_local.authorize(tool_name="send_email", action_description="")
    print(result2)
    # Beklenen: allowed=False, layer="structural"
    # (admin her şeyi yapabilir AMA localhost'ta send_email bloklu)

    print("\n=== TEST 3: admin rolü, production'da get_weather_forecast ===")
    admin_prod = PolicyService(role="admin", env="production")
    result3 = admin_prod.authorize(
        tool_name="get_weather_forecast",
        action_description="İzmir için yarının hava durumunu getir",
    )
    print(result3)
    # Beklenen: allowed=True (GEMINI_API_KEY yoksa semantic "skipped" notuyla)

    print("\n=== TEST 4: editor rolü, get_exam_results (yapısal İZİN) ===")
    editor_policy = PolicyService(role="editor", env="production")
    result4 = editor_policy.is_tool_allowed("get_exam_results")
    print(f"İzin var mı: {result4}")
    # Beklenen: True

    print("\n=== TEST 5 (DEBUG): Semantic check GERÇEKTEN API'ye gidiyor mu? ===")
    debug_policy = PolicyService(role="admin", env="production")
    print(f"GEMINI_API_KEY tanımlı mı: {bool(debug_policy.gemini_api_key)}")
    semantic_result = debug_policy.check_action_semantic(
        "Müşteriye şu mesajı gönder: TC kimlik numaram 12345678901, lütfen kaydet."
    )
    print(f"Semantic check sonucu: {semantic_result}")
    # Beklenen (API key varsa VE gerçekten çalışıyorsa):
    #   allowed=False olmalı, çünkü içerikte maskelenmemiş TC kimlik no var
    #   reason kısmında Gemini'nin kendi cümleleriyle "VIOLATION..." yazması beklenir
    # Eğer "Semantic check atlandı" yazıyorsa, key görülmüyor demektir.
