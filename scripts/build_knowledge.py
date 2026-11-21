"""Bangun app/data/knowledge.json dari paket tim DS (21 Nov 2026):
- output/dataset_pertanyaan/fakta_usaha.json  (fakta yang BOLEH disampaikan asisten)
- output/json/layanan.json                     (katalog 15 layanan S01–S15: kode, kategori, harga min/maks, durasi)
Ditambah aturan web dari aplikasi tim (halaman /pesan dan /cek-booking, aturan booking batal; keputusan tim 4 Nov 2026).
Nilai PERLU_DIISI_OWNER tidak ditulis sebagai fakta, tapi masuk daftar `tidak_diketahui`.
Pakai: python scripts/build_knowledge.py <folder output DS> [path keluaran]"""
import json, sys
from pathlib import Path
DS = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/capstone/ds_hasil_21nov/datasciencs-nail-art/output")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "app" / "data" / "knowledge.json"
F = json.loads((DS / "dataset_pertanyaan" / "fakta_usaha.json").read_text(encoding="utf-8"))
CAT = json.loads((DS / "json" / "layanan.json").read_text(encoding="utf-8"))
UNK = "PERLU_DIISI_OWNER"

ket = {h["layanan"]: h["keterangan"] for h in F["harga"]}
dur = {d["layanan"]: d["durasi_menit"] for d in F["durasi"]}
layanan = []
for s in CAT:
    if not s.get("aktif", True): continue
    n = s["nama_layanan"]
    assert n in ket and dur[n] == s["durasi_menit"], f"katalog & fakta tidak cocok: {n}"
    layanan.append({"kode": s["service_id"], "nama": n, "kategori": s["kategori"], "harga_mulai": s["harga_min"],
                    "harga_maks": None if s["harga_tetap"] else s["harga_maks"], "durasi_menit": s["durasi_menit"],
                    "catatan": ket[n]})
assert len(layanan) == 15

lok, bayar, kel = F["lokasi"], F["pembayaran"], F["perawatan_keluhan"]
ig = lok["instagram"]
tidak = []
if lok["alamat"] == UNK: tidak.append("alamat lengkap studio (yang diketahui hanya: Kota Jambi)")
if lok["patokan_parkir"] == UNK: tidak.append("patokan lokasi dan info parkir")
if UNK in kel["keluhan"]: tidak.append("nomor WhatsApp studio")
if bayar["rekening"] == UNK: tidak.append("nomor rekening untuk transfer")
if kel["garansi"] == UNK: tidak.append("garansi atau kebijakan perbaikan gratis setelah layanan")
tidak += ["promo atau diskon", "kebijakan keterlambatan datang", "biaya pembatalan",
          "hari libur mendatang selain yang tercatat (cek slot di halaman /pesan)",
          "layanan, desain khusus, atau harga di luar daftar layanan"]

kb = {
    "_catatan": ("Satu-satunya sumber fakta untuk asisten. DIBANGUN OTOMATIS oleh scripts/build_knowledge.py dari paket tim DS "
                 "(fakta_usaha.json + katalog layanan.json, 21 Nov 2026) ditambah aturan halaman web tim. Jangan edit manual: ubah sumbernya lalu jalankan ulang."),
    "versi": "2026-11-21",
    "sumber": ["DS fakta_usaha.json (katalog & jam operasional dikonfirmasi owner, interview 9 Okt 2026)", "DS output/json/layanan.json (15 layanan S01–S15)",
               "Aplikasi web tim: halaman /pesan dan /cek-booking (demo Arief 14 Nov 2026)", "Keputusan tim 4 Nov 2026: booking batal tidak diaktifkan lagi"],
    "studio": {"nama": "The Nail Palace", "deskripsi": "Studio nail art di Kota Jambi", "kota": lok["kota"], "instagram": ig},
    "layanan": layanan,
    "layanan_desain": {"kategori": F["layanan_desain"]["kategori"], "desain_custom": F["layanan_desain"]["desain_custom"],
                       "cakupan": F["layanan_desain"]["layanan_tidak_tersedia"]},
    "jam_operasional": F["jam_operasional"],
    "hari_libur_tercatat": F["hari_libur_tercatat"],
    "cara_reservasi": {
        "halaman_pesan": "/pesan", "tanpa_akun": "Reservasi tidak perlu membuat akun.",
        "langkah": F["cara_reservasi"][:4],
        "aturan": F["cara_reservasi"][4:] + ["Ketersediaan slot hanya bisa dilihat langsung di halaman /pesan (asisten tidak bisa mengecek slot)."],
        "kode_booking": "Setelah kirim, pelanggan mendapat kode booking berformat NP-YYYYMMDD-XXXX."},
    "cek_ubah_batal": {"halaman_cek": "/cek-booking",
                       "aturan": F["cek_ubah_batal"] + ["Reservasi yang sudah dibatalkan tidak bisa diaktifkan kembali; untuk pesan ulang buat booking baru (kode baru) dan tetap menunggu konfirmasi pemilik."]},
    "pembayaran": {"metode": bayar["metode"], "waktu": bayar["waktu"], "dp": bayar["dp_atau_bayar_online"]},
    "perawatan_keluhan": {"keluhan": "Keluhan hasil layanan disampaikan langsung ke studio (WhatsApp studio atau Instagram).",
                          "catatan_katalog": kel["catatan_katalog"]},
    "tidak_diketahui": tidak,
    "kontak": {"instagram": ig, "whatsapp": None},
    "arahan_kontak": f"Untuk info yang belum tersedia, sarankan pelanggan menghubungi studio The Nail Palace lewat Instagram {ig} (DM).",
}
OUT.write_text(json.dumps(kb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"OK {OUT} — {len(layanan)} layanan, {len(tidak)} info belum diketahui")
