"""Tes dengan Gemini di-mock (tanpa jaringan)."""
import httpx, pytest
from fastapi.testclient import TestClient
from app import main, genai, config

client = TestClient(main.app)
OK_BODY = {"candidates": [{"content": {"parts": [{"text": "Harga Gel Polish Polos Rp100.000."}]}}]}

class FakeHTTP:
    """Pengganti httpx.Client: mengembalikan urutan respons/eksepsi yang ditentukan."""
    def __init__(self, seq): self.seq, self.calls = list(seq), []
    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append({"url": url, "timeout": timeout, "system": json["systemInstruction"]["parts"][0]["text"]})
        item = self.seq.pop(0)
        if isinstance(item, Exception): raise item
        return httpx.Response(item[0], json=item[1], request=httpx.Request("POST", url))
    def close(self): pass

@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-dummy")
    monkeypatch.delenv("AE_INTERNAL_KEY", raising=False)
    monkeypatch.setattr(genai.time, "sleep", lambda s: None)

def use_fake(monkeypatch, seq):
    fake = FakeHTTP(seq)
    monkeypatch.setattr(genai.httpx, "Client", lambda: fake)
    return fake

# --- health & classify ---
def test_health_tidak_bocorkan_key():
    r = client.get("/health"); j = r.json()
    assert r.status_code == 200 and j["model_loaded"] and j["gemini_configured"] is True
    assert "test-key-dummy" not in r.text
    assert set(j["labels"]) == set(genai.ROUTES)  # 11 label dataset DS v1.0
    assert len(j["labels"]) == 11

def test_classify_bentuk_respons():
    r = client.post("/api/v1/classify", json={"pertanyaan": "Berapa harga gel polish?"}); j = r.json()
    assert r.status_code == 200
    assert j["label"] in j["probabilities"] and abs(sum(j["probabilities"].values()) - 1) < 1e-3
    assert j["confidence"] == max(j["probabilities"].values())
    assert j["yakin"] == (j["confidence"] >= config.CONF_THRESHOLD)
    assert j["aksi"] == (genai.aksi(j["label"]) if j["yakin"] else "ragu")

# --- validasi ---
@pytest.mark.parametrize("body,pesan", [
    ({"pertanyaan": ""}, "tidak boleh kosong"),
    ({"pertanyaan": "   \n "}, "tidak boleh kosong"),
    ({"pertanyaan": "a" * 501}, "terlalu panjang"),
    ({}, "field 'pertanyaan'"),
    ({"pertanyaan": 123}, "field 'pertanyaan'"),
])
@pytest.mark.parametrize("path", ["/api/v1/classify", "/api/v1/chat"])
def test_validasi(path, body, pesan):
    r = client.post(path, json=body)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "VALIDATION_ERROR" and pesan in r.json()["error"]["message"]

def test_validasi_bukan_json():
    r = client.post("/api/v1/classify", content=b"bukan json", headers={"Content-Type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "VALIDATION_ERROR"

def test_panjang_pas_500_diterima():
    assert client.post("/api/v1/classify", json={"pertanyaan": "a" * 500}).status_code == 200

def test_404_format_error():
    r = client.get("/tidak-ada"); assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"

# --- auth header ---
def test_auth_wajib_jika_env_diset(monkeypatch):
    monkeypatch.setenv("AE_INTERNAL_KEY", "rahasia-123")
    use_fake(monkeypatch, [(200, OK_BODY)])
    for path in ["/api/v1/classify", "/api/v1/chat"]:
        r = client.post(path, json={"pertanyaan": "harga gel polish?"})
        assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHORIZED"
        r = client.post(path, json={"pertanyaan": "harga gel polish?"}, headers={"X-Internal-Key": "salah"})
        assert r.status_code == 401
    r = client.post("/api/v1/classify", json={"pertanyaan": "harga gel polish?"}, headers={"X-Internal-Key": "rahasia-123"})
    assert r.status_code == 200
    r = client.post("/api/v1/chat", json={"pertanyaan": "harga gel polish?"}, headers={"X-Internal-Key": "rahasia-123"})
    assert r.status_code == 200 and r.json()["sumber"] == "gemini"
    assert client.get("/health").status_code == 200  # health tetap terbuka (untuk Render)

def test_tanpa_env_key_boleh(monkeypatch):
    assert client.post("/api/v1/classify", json={"pertanyaan": "jam buka?"}).status_code == 200

# --- Gemini robustness ---
def test_chat_sukses(monkeypatch):
    fake = use_fake(monkeypatch, [(200, OK_BODY)])
    j = client.post("/api/v1/chat", json={"pertanyaan": "Berapa harga gel polish?"}).json()
    assert j["sumber"] == "gemini" and j["model_dipakai"] == config.GEMINI_PRIMARY and j["percobaan"] == 1
    assert j["jawaban"] == "Harga Gel Polish Polos Rp100.000."
    assert "Rp100.000" in fake.calls[0]["system"] and "Paket Wisuda / Bridal" in fake.calls[0]["system"]  # fakta knowledge.json (15 layanan)
    assert j["aksi"] in set(genai.AKSI_INSTRUKSI) | {"ragu"}
    assert fake.calls[0]["timeout"] <= config.GEMINI_TIMEOUT

def test_503_lalu_sukses(monkeypatch):
    fake = use_fake(monkeypatch, [(503, {"error": "busy"}), (200, OK_BODY)])
    j = client.post("/api/v1/chat", json={"pertanyaan": "Berapa harga gel polish?"}).json()
    assert j["sumber"] == "gemini" and j["model_dipakai"] == config.GEMINI_PRIMARY and j["percobaan"] == 2
    assert all(config.GEMINI_PRIMARY in c["url"] for c in fake.calls)

def test_timeout_lalu_model_cadangan(monkeypatch):
    fake = use_fake(monkeypatch, [httpx.ReadTimeout("t"), (503, {}), (200, OK_BODY)])
    j = client.post("/api/v1/chat", json={"pertanyaan": "Berapa harga gel polish?"}).json()
    assert j["sumber"] == "gemini" and j["model_dipakai"] == config.GEMINI_BACKUP and j["percobaan"] == 3
    assert config.GEMINI_BACKUP in fake.calls[-1]["url"]

@pytest.mark.parametrize("q,label_kata", [
    ("Berapa harga gel polish?", "/pesan"),
    ("Cara booking gimana kak?", "/pesan"),
])
def test_semua_gagal_fallback(monkeypatch, q, label_kata):
    use_fake(monkeypatch, [(503, {}), httpx.ConnectError("x"), (500, {})])
    r = client.post("/api/v1/chat", json={"pertanyaan": q}); j = r.json()
    assert r.status_code == 200
    assert j["sumber"] == "fallback" and j["model_dipakai"] is None and j["percobaan"] == 3
    assert label_kata in j["jawaban"] and "sibuk" in j["jawaban"]

def test_400_tidak_diulang(monkeypatch):
    fake = use_fake(monkeypatch, [(400, {}), (200, OK_BODY)])
    j = client.post("/api/v1/chat", json={"pertanyaan": "harga gel polish?"}).json()
    assert j["sumber"] == "fallback" and len(fake.calls) == 1

def test_respons_kosong_dianggap_gagal(monkeypatch):
    use_fake(monkeypatch, [(200, {"candidates": []}), (200, {"candidates": [{"content": {"parts": [{"text": ""}]}}]}), (200, OK_BODY)])
    j = client.post("/api/v1/chat", json={"pertanyaan": "harga gel polish?"}).json()
    assert j["sumber"] == "gemini" and j["model_dipakai"] == config.GEMINI_BACKUP

def test_tanpa_gemini_key_langsung_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY"); monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    fake = use_fake(monkeypatch, [])
    j = client.post("/api/v1/chat", json={"pertanyaan": "jam buka?"}).json()
    assert j["sumber"] == "fallback" and fake.calls == []
    assert client.get("/health").json()["gemini_configured"] is False

def test_label_ragu_diberi_tahu_ke_prompt(monkeypatch):
    fake = use_fake(monkeypatch, [(200, OK_BODY)])
    monkeypatch.setattr(config, "CONF_THRESHOLD", 1.01)  # paksa semua dianggap ragu
    j = client.post("/api/v1/chat", json={"pertanyaan": "halo"}).json()
    assert j["yakin"] is False and "TIDAK YAKIN" in fake.calls[0]["system"]

def test_label_yakin_tanpa_catatan(monkeypatch):
    fake = use_fake(monkeypatch, [(200, OK_BODY)])
    monkeypatch.setattr(config, "CONF_THRESHOLD", 0.0)
    client.post("/api/v1/chat", json={"pertanyaan": "halo"})
    assert "TIDAK YAKIN" not in fake.calls[0]["system"]

def test_fallback_semua_label_dari_knowledge():
    kb = genai.load_kb()
    for lab in genai.ROUTES:
        t = genai.fallback_answer(lab, kb)
        assert t and "{" not in t
    assert "/cek-booking" in genai.fallback_answer("cara_reservasi", kb)
    assert "/cek-booking" in genai.fallback_answer("cek_ubah_batal", kb)
    assert "Rp60.000" in genai.fallback_answer("harga", kb)
    assert "10:00–19:00" in genai.fallback_answer("jam_operasional", kb) and "Minggu" in genai.fallback_answer("jam_operasional", kb)
    assert "QRIS" in genai.fallback_answer("pembayaran", kb)
    assert "instagram.com" in genai.fallback_answer("lokasi", kb)

def test_routing_11_label_lengkap():
    assert len(genai.ROUTES) == 11
    assert set(genai.ROUTES) == set(main.STATE["clf"].labels)  # setiap label model punya rute
    assert {genai.aksi(l) for l in genai.ROUTES} == set(genai.AKSI_INSTRUKSI)
    assert genai.aksi("label_tak_dikenal") == "arahkan_ke_kontak"
    kb = genai.load_kb()
    for lab in genai.ROUTES:  # prompt bisa dirakit untuk semua label, berisi arahan aksinya
        p = genai.build_system_prompt(lab, True, kb)
        assert genai.AKSI_INSTRUKSI[genai.aksi(lab)] in p and "{" not in p.split("## FAKTA")[0]

def test_knowledge_15_layanan_tanpa_placeholder():
    kb = genai.load_kb(); txt = genai.config.KNOWLEDGE_PATH.read_text()
    assert len(kb["layanan"]) == 15 and [s["kode"] for s in kb["layanan"]] == [f"S{i:02d}" for i in range(1, 16)]
    assert "PERLU_DIISI_OWNER" not in txt

def test_log_tidak_memuat_pertanyaan(monkeypatch, caplog):
    use_fake(monkeypatch, [(503, {}), (503, {}), (503, {})])
    rahasia = "nama saya Siti 081234567890 mau batal"
    with caplog.at_level("INFO"):
        client.post("/api/v1/chat", json={"pertanyaan": rahasia})
    assert "081234567890" not in caplog.text and "Siti" not in caplog.text

def test_fallback_label_ragu_umum():
    kb = genai.load_kb(); t = genai.fallback_answer("cara_reservasi", kb, yakin=False)
    assert "/pesan" in t and "/cek-booking" in t and "hubungi studio" in t

def test_404_langsung_ke_cadangan(monkeypatch):
    fake = use_fake(monkeypatch, [(404, {}), (200, OK_BODY)])
    j = client.post("/api/v1/chat", json={"pertanyaan": "harga gel polish?"}).json()
    assert j["sumber"] == "gemini" and j["model_dipakai"] == config.GEMINI_BACKUP and len(fake.calls) == 2

def test_fallback_tanpa_kalimat_instruksi_bot():
    kb = genai.load_kb()
    for lab in genai.ROUTES:
        assert "sarankan pelanggan" not in genai.fallback_answer(lab, kb)
