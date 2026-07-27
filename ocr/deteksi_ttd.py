#!/usr/bin/env python3
"""
deteksi_ttd.py
Deteksi ADA/TIDAK tanda tangan di ujung kanan-bawah halaman ijazah.
Metode: RULE-BASED (tanpa machine learning).
  1. crop ROI kanan-bawah
  2. binarisasi (Otsu) -> tinta jadi foreground
  3. buang garis lurus panjang (garis tanda tangan / bingkai tabel)
  4. analisis komponen terhubung + kepadatan tinta -> keputusan heuristik

Contoh pakai:
    python deteksi_ttd.py ijazah.pdf
    python deteksi_ttd.py ijazah.jpg --debug
    python deteksi_ttd.py ijazah.png --roi 0.5 0.6 1.0 1.0

Dependensi:
    pip install opencv-python-headless numpy pillow pdf2image
    sistem: poppler-utils (untuk PDF)
"""
import argparse, os, sys
import numpy as np
import cv2

# ---------- MEMUAT BERKAS (pdf / jpg / png) ----------
def load_image(path, page=0, dpi=300):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        from pdf2image import convert_from_path
        pages = convert_from_path(path, dpi=dpi)
        if page >= len(pages):
            raise SystemExit(f"PDF hanya punya {len(pages)} halaman.")
        return cv2.cvtColor(np.array(pages[page]), cv2.COLOR_RGB2BGR)
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Gagal membaca gambar: {path}")
    return img

def crop_roi(img, roi):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0*h):int(y1*h), int(x0*w):int(x1*w)].copy()

# ---------- ANALISIS TANDA TANGAN (rule-based) ----------
def analisis_ttd(roi_bgr, min_ink=0.004, max_ink=0.25):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    Hroi, Wroi = gray.shape
    luas_roi = Hroi * Wroi

    # binarisasi: tinta (gelap) -> putih (foreground=255)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # buang garis lurus panjang (garis alas ttd / bingkai) supaya tak terhitung
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
             cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, Wroi//12), 1)))
    vert  = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
             cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, Hroi//12))))
    tanpa_garis = cv2.subtract(bw, cv2.bitwise_or(horiz, vert))

    # sambungkan coretan yang terputus
    tanpa_garis = cv2.morphologyEx(tanpa_garis, cv2.MORPH_CLOSE,
             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    ink_ratio = float(np.count_nonzero(tanpa_garis)) / luas_roi

    # komponen terhubung
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(tanpa_garis, connectivity=8)
    min_area = max(30, luas_roi * 0.0004)          # abaikan bintik noise
    idx = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]

    blob = dict(bbox=None, komponen=0, ink_ratio=ink_ratio,
                tinggi_teks=0, h_ratio=0, area_frac=0, extent=0, width_frac=0)
    verdict, score = False, 0.0

    if idx:
        heights = np.array([stats[i, cv2.CC_STAT_HEIGHT] for i in idx])
        # perkiraan tinggi TEKS CETAK = median tinggi komponen
        # (teks cetak = banyak komponen kecil seragam; tanda tangan = 1 komponen menonjol)
        tinggi_teks = float(np.median(heights))

        # kandidat tanda tangan = komponen dengan AREA terbesar
        cand = max(idx, key=lambda i: stats[i, cv2.CC_STAT_AREA])
        cx, cy = stats[cand, cv2.CC_STAT_LEFT], stats[cand, cv2.CC_STAT_TOP]
        cw, ch = stats[cand, cv2.CC_STAT_WIDTH], stats[cand, cv2.CC_STAT_HEIGHT]
        carea  = int(stats[cand, cv2.CC_STAT_AREA])

        h_ratio    = ch / max(1.0, tinggi_teks)     # tanda tangan jauh lebih tinggi dari teks
        area_frac  = carea / luas_roi
        extent     = carea / max(1, cw*ch)           # strokey -> kecil
        width_frac = cw / Wroi

        blob.update(bbox=(cx,cy,cx+cw,cy+ch), komponen=len(idx),
                    tinggi_teks=tinggi_teks, h_ratio=h_ratio, area_frac=area_frac,
                    extent=extent, width_frac=width_frac)

        # ---- aturan keputusan (heuristik, silakan disetel) ----
        c_ink    = min_ink <= ink_ratio <= max_ink   # ada tinta, tidak penuh
        c_tall   = h_ratio >= 1.8                     # komponen jauh lebih tinggi dari teks cetak
        c_area   = area_frac >= 0.006                 # gumpalan coretan cukup besar
        c_stroke = extent <= 0.45                     # strokey, bukan blok/kotak pejal
        c_width  = width_frac >= 0.12                 # melebar (khas goresan tangan)
        score = float(np.mean([c_ink, c_tall, c_area, c_stroke, c_width]))
        # syarat wajib: menonjol lebih tinggi dari teks, strokey, ada tinta wajar,
        # dan cukup besar/lebar
        verdict = bool(c_tall and c_stroke and c_ink and (c_area or c_width))

    return verdict, score, blob, dict(bw=bw, tanpa_garis=tanpa_garis)

# ---------- MAIN (CLI) ----------
def main():
    ap = argparse.ArgumentParser(description="Deteksi ada/tidaknya tanda tangan (rule-based, tanpa ML).")
    ap.add_argument("berkas", help="berkas ijazah (.pdf/.jpg/.png)")
    ap.add_argument("--roi", nargs=4, type=float, default=[0.5, 0.58, 1.0, 1.0],
                    metavar=("X0","Y0","X1","Y1"),
                    help="wilayah kanan-bawah dalam fraksi 0-1 (default: 0.5 0.58 1.0 1.0)")
    ap.add_argument("--page", type=int, default=0, help="halaman PDF (0=pertama)")
    ap.add_argument("--dpi", type=int, default=300, help="resolusi render PDF")
    ap.add_argument("--debug", action="store_true", help="simpan ROI, mask, & anotasi")
    args = ap.parse_args()

    if not os.path.exists(args.berkas):
        raise SystemExit(f"Berkas tidak ditemukan: {args.berkas}")

    img = load_image(args.berkas, page=args.page, dpi=args.dpi)
    roi = crop_roi(img, args.roi)
    verdict, score, b, dbg = analisis_ttd(roi)

    print("=" * 60)
    print(f"Berkas       : {args.berkas}")
    print(f"Ukuran       : {img.shape[1]} x {img.shape[0]} px | ROI kanan-bawah: {args.roi}")
    print("-" * 60)
    print("Fitur (rule-based, tanpa ML):")
    print(f"  Kepadatan tinta        : {b['ink_ratio']*100:6.3f} %")
    print(f"  Jumlah komponen        : {b['komponen']}")
    print(f"  Tinggi teks cetak (med): {b['tinggi_teks']:.0f} px")
    print(f"  Komponen terbesar       :")
    print(f"    - tinggi vs teks (h_ratio): {b['h_ratio']:.2f}  (>=1.8 = menonjol seperti coretan)")
    print(f"    - luas thd ROI (area_frac): {b['area_frac']*100:6.3f} %")
    print(f"    - extent (kestrokean)     : {b['extent']:.3f}  (kecil = coretan)")
    print(f"    - lebar relatif           : {b['width_frac']:.3f}")
    print("-" * 60)
    print(f"Skor keyakinan : {score:.2f}")
    print("HASIL          : " + ("ADA tanda tangan terdeteksi." if verdict
                                  else "TIDAK terdeteksi tanda tangan."))
    print("=" * 60)

    if args.debug:
        base = os.path.splitext(os.path.basename(args.berkas))[0]
        cv2.imwrite(f"debug_{base}_ttd_roi.png", roi)
        cv2.imwrite(f"debug_{base}_ttd_biner.png", dbg["bw"])
        cv2.imwrite(f"debug_{base}_ttd_tanpa_garis.png", dbg["tanpa_garis"])
        anot = img.copy(); h,w = img.shape[:2]; x0,y0,x1,y1 = args.roi
        cv2.rectangle(anot,(int(x0*w),int(y0*h)),(int(x1*w),int(y1*h)),(0,180,255),4)
        if b["bbox"]:
            xs,ys,xe,ye = b["bbox"]; ox,oy = int(x0*w),int(y0*h)
            col = (0,200,0) if verdict else (0,0,255)
            cv2.rectangle(anot,(ox+xs,oy+ys),(ox+xe,oy+ye),col,4)
        cv2.imwrite(f"debug_{base}_ttd_anotasi.png", anot)
        print(f"[debug] disimpan: debug_{base}_ttd_roi.png, _biner.png, _tanpa_garis.png, _anotasi.png")

    sys.exit(0 if verdict else 1)

if __name__ == "__main__":
    main()
