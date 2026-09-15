https://pypi.org/project/pytesseract/
https://pytesseract.readthedocs.io/en/latest/

Semua siap (Tesseract + bahasa `ind` sudah terpasang). Saya tulis dua skrip CLI, lalu uji dengan contoh ijazah sintetis (karena saya tak punya ijazah asli).Sekarang saya buat contoh ijazah sintetis (dengan & tanpa tanda tangan, format PNG & PDF) untuk menguji kedua skrip.
OCR-nya sudah membaca `NOMOR : DN-12 Dd 0123456` dengan benar — tapi logika saya terlalu ketat (menuntut kata "NOMOR" dan "IJAZAH" berdampingan). 
Pada ijazah asli, di kanan-atas biasanya hanya tertulis label **"NOMOR"** diikuti kodenya, sedang kata "IJAZAH" adalah judul di tengah. Saya perbaiki agar deteksinya realistis: cocokkan label "NOMOR"/"NOMOR IJAZAH" **atau** pola nomor itu sendiri.Deteksi nomor ijazah bekerja: `DN-12 DD 0123456`. 

Sekarang uji deteksi tanda tangan pada kedua kasus (ada & tidak ada).False positive: versi tanpa tanda tangan pun terdeteksi "ADA", karena ROI kanan-bawah berisi teks cetak ("Kepala Sekolah", nama, NIP) yang ikut terhitung. Diskriminatornya harus membedakan **coretan tangan** dari **teks cetak**. 

Kunci rule-based-nya: tanda tangan adalah satu komponen terhubung yang **jauh lebih tinggi/besar** daripada baris teks cetak. Saya perbaiki.Sekarang keduanya benar: **ada TTD → h_ratio 7,38 → "ADA"**; **tanpa TTD → h_ratio 1,00 → "TIDAK"**. Uji cepat format PNG & JPEG juga.Semua format (PDF/PNG/JPEG) jalan. Saya lihat gambar debug beranotasi untuk memastikan kotak deteksi tepat.Kedua skrip terbukti jalan pada PDF/PNG/JPEG, untuk kasus positif maupun negatif. (Penampil gambar sedang bermasalah, tapi metrik & exit code sudah mengonfirmasi hasilnya.) Ini file-nya.Dua skrip CLI, sudah saya uji nyata pada ijazah sintetis (PDF/PNG/JPEG, kasus ada & tanpa TTD).

**1. `deteksi_nomor_ijazah.py`** — OCR di ROI kanan-atas.
Alur: muat berkas → crop kanan-atas → pra-pemrosesan (perbesar, denoise, Otsu) → OCR Tesseract → cocokkan label & ekstrak kode.
Hasil uji: membaca `NOMOR : DN-12 Dd 0123456` → **"NOMOR IJAZAH TERDETEKSI"**, nomor `DN-12 DD 0123456`.
Catatan penting: di ijazah asli, kanan-atas biasanya cuma tertulis **"NOMOR"** + kode (kata "IJAZAH" jadi judul di tengah), jadi skripnya menerima tiga cara cocok: frasa lengkap, label "NOMOR"+kode, atau pola kode khas.

**2. `deteksi_ttd.py`** — rule-based, **tanpa ML**, di ROI kanan-bawah.
Alur: crop kanan-bawah → binarisasi Otsu → **buang garis lurus panjang** (garis alas/bingkai) → komponen terhubung → keputusan heuristik.
Kunci membedakan tanda tangan dari teks cetak: **komponen terbesar jauh lebih tinggi dari baris teks cetak** (`h_ratio`), strokey (`extent` kecil), dan melebar. Hasil uji: ada TTD → h_ratio 7,4 → **"ADA"**; tanpa TTD → h_ratio 1,0 → **"TIDAK"**.

**Cara pakai:**
```bash
python deteksi_nomor_ijazah.py ijazah.pdf
python deteksi_ttd.py ijazah.jpg --debug
# opsi: --roi X0 Y0 X1 Y1  (fraksi 0-1), --page, --dpi, --debug
```
Keduanya menerima `.pdf/.jpg/.png`, mengembalikan exit code 0 (terdeteksi) / 1 (tidak) supaya bisa dipipa di shell. `--debug` menyimpan crop ROI, mask biner, dan gambar beranotasi.

**Dependensi:**
```bash
pip install opencv-python-headless numpy pillow pytesseract pdf2image
# sistem (Ubuntu/WSL):
sudo apt install tesseract-ocr tesseract-ocr-ind poppler-utils
```

Dua hal jujur soal keterbatasannya:
- **ROI & ambang perlu disetel** untuk template ijazahmu yang asli. Default-ku mengasumsikan pindaian tegak; jalankan `--debug` dulu untuk melihat apakah kotak ROI mengenai area yang benar, lalu geser `--roi`.
- Deteksi TTD murni heuristik — bisa keliru bila ada stempel/logo besar di kanan-bawah atau tanda tangan sangat kecil. Semua fitur dicetak agar mudah kamu kalibrasi (mis. turunkan ambang `h_ratio` bila TTD tipis).

Kalau mau, saya bisa gabungkan keduanya jadi **satu CLI** (`analisis_ijazah.py`) yang sekaligus cek nomor, verifikasi frasa "IJAZAH SMA/Madrasah", dan deteksi TTD, lalu keluarkan ringkasan JSON — cocok untuk jadi kerangka proyek akhir mahasiswa.
---

## Pra-pemrosesan bila halaman harus diputar dulu (kasus `contoh_ijazah/ijazah_s2.jpg`)

ROI pada kedua skrip ditulis sebagai **fraksi halaman** (mis. `0.55 0.0 1.0 0.28` = kanan-atas).
Fraksi itu baru bermakna kalau halamannya sudah tegak. Pindaian `ijazah_s2.jpg` tersimpan
**terbaring 90°** (2481×3506, teks terbaca dari bawah ke atas), jadi "kanan-atas" jatuh ke
bagian halaman yang salah **dan** huruf-hurufnya terbaring — Tesseract hanya sanggup
mengenali huruf yang mendatar. Karena itu koreksi geometri harus dikerjakan **sebelum**
`crop_roi()`, bukan sesudahnya.

### Rantai pra-pemrosesan yang dibutuhkan (urutannya penting)

| # | Tahap | Tujuan | Fungsi di `preprocess.py` |
|---|-------|--------|---------------------------|
| 1 | **EXIF Orientation** | `cv2.imread()` mengabaikan tag EXIF, sehingga foto yang di HP terlihat tegak bisa termuat terbaring | `terapkan_exif()` |
| 2 | **Orientasi kasar 0/90/180/270** | menegakkan halaman; tanpa ini semua tahap berikutnya sia-sia | `deteksi_orientasi()` + `putar_kelipatan_90()` |
| 3 | **Deskew halus ±15°** | membuat baris teks benar-benar mendatar; miring 1–2° saja sudah menurunkan akurasi OCR | `estimasi_skew()` + `luruskan()` |
| 4 | *(opsional)* **Potong tepi / perspektif** | untuk **foto HP**, bukan hasil scan: membuang latar meja & meluruskan lembar yang terfoto miring | `potong_dokumen()` |
| 5 | **Normalisasi iluminasi** | meratakan kertas krem + watermark UI agar Otsu stabil | `normalisasi_iluminasi()` |
| → | baru **crop ROI** lalu pra-pemrosesan per-ROI yang sudah ada (perbesar 2×, denoise, Otsu) | | |

### Cara menebak orientasi (tahap 2)

Dipakai dua metode, yang pertama jadi andalan dan yang kedua jadi cadangan:

1. **Tesseract OSD** (`image_to_osd`, `--psm 0`) — butuh data bahasa `osd`
   (`sudo apt install tesseract-ocr-osd`). Ini **satu-satunya cara andal membedakan 0° vs 180°**,
   karena keduanya sama-sama punya baris teks mendatar. Pada `ijazah_s2.jpg` OSD menjawab
   `Rotate: 90`. Keyakinannya rendah (±20) karena latar kertas bertekstur, jadi jangan
   dijadikan syarat mutlak.
2. **Profil proyeksi** (cadangan, tanpa OCR) — teks tegak membuat profil jumlah piksel
   **per-baris** naik-turun tajam (baris gelap diselingi celah putih); kalau halaman terbaring,
   justru profil **per-kolom** yang tajam. Bandingkan ketajaman keduanya
   (jumlah kuadrat selisih antar-elemen). Metode ini hanya menjawab *"perlu diputar 90° atau tidak"*,
   **tidak bisa** membedakan 0° vs 180° maupun 90° vs 270°.

Putarannya memakai `cv2.rotate()` (kelipatan 90°, **tanpa interpolasi** → tidak ada piksel yang
rusak), bukan `warpAffine`.

### Cara mengukur skew (tahap 3)

Tebakan kasar dari **median sudut `minAreaRect`** atas gumpalan baris teks (huruf disatukan
dulu dengan dilasi memanjang 25×5); median dipakai supaya tahan baris pencilan. Tebakan itu
lalu dihaluskan dengan **pencarian profil proyeksi** ±1.5° langkah 0.1°: sudut terbaik adalah
yang membuat profil per-baris paling "bergerigi". `luruskan()` memperbesar kanvas agar sudut
halaman tidak terpotong dan mengisi sisa dengan putih.

### Pemakaian

```bash
# lihat hasil pra-pemrosesannya saja
python preprocess.py contoh_ijazah/ijazah_s2.jpg --debug

# kedua skrip deteksi: cukup tambahkan --auto-rotate
python deteksi_nomor_ijazah.py contoh_ijazah/ijazah_s2.jpg --auto-rotate --roi 0.02 0.80 0.45 0.98
python deteksi_ttd.py          contoh_ijazah/ijazah_s2.jpg --auto-rotate --roi 0.62 0.60 1.00 0.92

# opsi tambahan: --tanpa-osd (paksa metode proyeksi), --crop-dokumen (foto HP)
```

### Hasil uji pada `ijazah_s2.jpg`

```
[pra-proses] putar 90d (osd, conf 20.49), skew 0.0d, exif=None, crop=False
Ukuran      : 2481 x 3506  ->  3506 x 2481
Teks ROI    : "Prof. Ari Kuncoro, S.E., M.A., Ph.D. / Nomor yjazah: 571012022000056"
HASIL       : NOMOR IJAZAH TERDETEKSI  -> 571012022000056   (cocok dengan aslinya)
TTD (dekan) : h_ratio 8.71 -> ADA
```

Tanpa `--auto-rotate`, ROI yang sama hanya menghasilkan sampah
(`"PTN YA INA LIL BAN ALAN BAPER RENA"`) dan **false positive** `TAN 252. S`.

Uji putar sintetis 0/90/180/270° pada berkas ini: keempatnya dikembalikan ke tegak
persis (selisih piksel = 0).

### Catatan

- **Skew ≠ 0 pada berkas lain.** `cth-ijazah.jpg` terkoreksi −0.3°, `cth-ijazah-2.jpg` −1.46°,
  keduanya orientasi 0°. Jadi `--auto-rotate` aman dipakai untuk semua berkas, bukan hanya yang terbaring.
- **ROI tetap harus disetel per-template.** Menegakkan halaman tidak memindahkan ROI:
  pada ijazah UI ini nomor ada di **kiri-bawah** (bukan kanan-atas seperti default skrip),
  dan barcode `NC. 21-006359` di kanan-bawah. Jalankan `--debug` untuk memastikan kotaknya tepat.
- Nama berkas di repo ini `ijazah_s2.**jpg**` (bukan `.png`).

<!-- POSISI IJAZAH -->
(TOP-RIGHT) KANAN-ATAS = 0.5 0.0 1.0 0.30
(BOTTOM-LEFT) KIRI-BAWAH = 0.02 0.80 0.45 0.98
(BOTTOM-RIGHT) KANAN-BAWAH = 0.62 0.60 1.00 0.92

The ROI is given as four fractions of the page, --roi x0 y0 x1 y1, where (x0, y0) is
the top-left corner and (x1, y1) the bottom-right, the origin is the top-left of the page, and y grows downward. So the default 0.55 0.0 1.0 0.28 means “the right 45% of the width, top 28% of the height”.


<!-- ijazah UI -->
python deteksi_nomor_ijazah.py contoh_ijazah/ijazah_s2.jpg --auto-rotate --roi 0.02 0.80 0.45 0.98 --debug
python deteksi_ttd.py          contoh_ijazah/ijazah_s2.jpg --auto-rotate --roi 0.62 0.60 1.00 0.92