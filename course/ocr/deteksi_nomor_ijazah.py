#!/usr/bin/env python3
"""
deteksi_nomor_ijazah.py
Deteksi kata 'NOMOR IJAZAH' di ujung kanan-atas halaman ijazah, lalu baca nomornya.
Metode: crop wilayah kanan-atas (ROI) -> pra-pemrosesan -> OCR (Tesseract).

Contoh pakai:
    python deteksi_nomor_ijazah.py ijazah.pdf
    python deteksi_nomor_ijazah.py ijazah.jpg --debug
    python deteksi_nomor_ijazah.py ijazah.png --roi 0.5 0.0 1.0 0.30 --dpi 300

Dependensi:
    pip install opencv-python-headless numpy pillow pytesseract pdf2image
    sistem: tesseract-ocr (+ bahasa: tesseract-ocr-ind), poppler-utils (untuk PDF)
"""
import argparse, os, re, sys
import numpy as np
import cv2

# ---------- 1. MEMUAT BERKAS (pdf / jpg / png) ----------
def load_image(path, page=0, dpi=300):
    """Kembalikan citra BGR (numpy) dari pdf/jpg/png."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pdf2image import convert_from_path
        pages = convert_from_path(path, dpi=dpi)
        if page >= len(pages):
            raise SystemExit(f"PDF hanya punya {len(pages)} halaman (minta halaman {page}).")
        rgb = np.array(pages[page])                    # RGB
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Gagal membaca berkas gambar: {path}")
    return img

# ---------- 2. AMBIL ROI KANAN-ATAS ----------
def crop_roi(img, roi):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0*h):int(y1*h), int(x0*w):int(x1*w)].copy(), (int(x0*w), int(y0*h))

# ---------- 3. PRA-PEMROSESAN UNTUK OCR ----------
def preprocess_for_ocr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # perbesar agar huruf kecil lebih terbaca
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 5, 40, 40)               # reduksi noise, jaga tepi
    # binarisasi Otsu (teks gelap di latar terang)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw

# ---------- 4. OCR ----------
def run_ocr(bw, lang_pref="ind"):
    import pytesseract
    for lang in (lang_pref, "eng"):
        try:
            txt = pytesseract.image_to_string(bw, lang=lang, config="--psm 6")
            return txt, lang
        except pytesseract.TesseractError:
            continue
    return pytesseract.image_to_string(bw, config="--psm 6"), "default"

# ---------- 5. CARI LABEL & EKSTRAK NOMOR ----------
def cari_nomor_ijazah(teks):
    """
    Di ijazah asli, kanan-atas umumnya tertulis label 'NOMOR' + kode
    (kata 'IJAZAH' sering jadi judul di tengah, bukan di sebelah nomor).
    Jadi 'ketemu' bila salah satu terpenuhi:
      (a) frasa lengkap 'NOMOR IJAZAH' / 'NO IJAZAH', atau
      (b) label 'NOMOR'/'NO' diikuti pola kode, atau
      (c) pola nomor ijazah yang khas muncul di ROI kanan-atas.
    """
    norm = re.sub(r"[ \t]+", " ", teks.upper())
    lines = [l.strip() for l in norm.splitlines() if l.strip()]

    kw_full  = re.compile(r"\bNO(?:MOR)?\.?\s*(?:INDUK\s*)?IJAZAH\b")
    kw_label = re.compile(r"\bNO(?:MOR)?\b\.?\s*:?")
    # pola kode ijazah (mis. DN-12 Dd 0123456, MA-xxxxxxx, atau deret panjang)
    pola = re.compile(r"([A-Z]{1,4}[-/ ]?\d[A-Z0-9][A-Z0-9\-/ .]{4,})|(\d{6,})")

    def ambil_kode(s):
        m = pola.search(s.replace(":", " "))
        return m.group(0).strip(" :.-") if m else None

    # (a) frasa lengkap
    for i, line in enumerate(lines):
        if kw_full.search(line):
            kode = ambil_kode(kw_full.sub("", line)) or ambil_kode(lines[i+1] if i+1 < len(lines) else "")
            return True, kode, "frasa 'NOMOR IJAZAH'"

    # (b) label NOMOR/NO + kode pada baris yang sama
    for line in lines:
        if kw_label.search(line):
            kode = ambil_kode(kw_label.sub("", line, count=1))
            if kode:
                return True, kode, "label 'NOMOR' + kode"

    # (c) pola kode khas di ROI (tanpa label terbaca)
    kode = ambil_kode(norm)
    if kode:
        return True, kode, "pola nomor di ROI kanan-atas"

    return False, None, None

# ---------- MAIN (CLI) ----------
def main():
    ap = argparse.ArgumentParser(description="Deteksi 'NOMOR IJAZAH' di ujung kanan-atas ijazah.")
    ap.add_argument("berkas", help="berkas ijazah (.pdf/.jpg/.png)")
    ap.add_argument("--roi", nargs=4, type=float, default=[0.55, 0.0, 1.0, 0.28],
                    metavar=("X0","Y0","X1","Y1"),
                    help="wilayah kanan-atas dalam fraksi 0-1 (default: 0.55 0.0 1.0 0.28)")
    ap.add_argument("--page", type=int, default=0, help="halaman PDF (0=pertama)")
    ap.add_argument("--dpi", type=int, default=300, help="resolusi render PDF")
    ap.add_argument("--lang", default="ind", help="bahasa Tesseract (default ind, fallback eng)")
    ap.add_argument("--debug", action="store_true", help="simpan crop ROI & gambar beranotasi")
    args = ap.parse_args()

    if not os.path.exists(args.berkas):
        raise SystemExit(f"Berkas tidak ditemukan: {args.berkas}")

    img = load_image(args.berkas, page=args.page, dpi=args.dpi)
    roi, (ox, oy) = crop_roi(img, args.roi)
    bw = preprocess_for_ocr(roi)
    teks, lang = run_ocr(bw, args.lang)
    ketemu, nomor, cara = cari_nomor_ijazah(teks)

    print("=" * 60)
    print(f"Berkas      : {args.berkas}")
    print(f"Ukuran      : {img.shape[1]} x {img.shape[0]} px  | ROI kanan-atas: {args.roi}")
    print(f"Bahasa OCR  : {lang}")
    print("-" * 60)
    print("Teks terbaca di ROI:")
    print("  " + "\n  ".join([l for l in teks.splitlines() if l.strip()]) or "  (kosong)")
    print("-" * 60)
    if ketemu:
        print("HASIL: NOMOR IJAZAH TERDETEKSI di ujung kanan-atas.")
        print(f"Cara cocok             : {cara}")
        print(f"Nomor ijazah (perkiraan): {nomor if nomor else '(label terbaca, kode kurang jelas)'}")
    else:
        print("HASIL: nomor ijazah TIDAK terdeteksi di ROI.")
        print("Saran: sesuaikan --roi atau naikkan --dpi.")
    print("=" * 60)

    if args.debug:
        pathdirhasil = "hasil"
        base = os.path.splitext(os.path.basename(args.berkas))[0]
        cv2.imwrite(f"{pathdirhasil}/debug_{base}_roi.png", roi)
        anot = img.copy()
        h, w = img.shape[:2]
        x0,y0,x1,y1 = args.roi
        cv2.rectangle(anot,(int(x0*w),int(y0*h)),(int(x1*w),int(y1*h)),(0,180,255),4)
        cv2.imwrite(f"{pathdirhasil}/debug_{base}_anotasi.png", anot)
        print(f"[debug] disimpan: debug_{base}_roi.png, debug_{base}_anotasi.png")

    sys.exit(0 if ketemu else 1)

if __name__ == "__main__":
    main()
