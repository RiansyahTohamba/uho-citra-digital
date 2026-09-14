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