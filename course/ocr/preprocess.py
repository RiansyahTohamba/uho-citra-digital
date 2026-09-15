#!/usr/bin/env python3
"""
preprocess.py
Pra-pemrosesan GEOMETRI halaman ijazah: menegakkan citra SEBELUM ROI diambil.

Kenapa perlu?
    ROI pada deteksi_nomor_ijazah.py / deteksi_ttd.py ditulis sebagai *fraksi*
    halaman (mis. 0.55 0.0 1.0 0.28 = kanan-atas). Fraksi itu hanya bermakna
    kalau halamannya sudah tegak. Pada berkas seperti contoh_ijazah/ijazah_s2.jpg
    hasil pindaian tersimpan miring 90 derajat, sehingga "kanan-atas" jatuh ke
    bagian halaman yang salah dan OCR membaca huruf yang terbaring.

Urutan yang dipakai (urutannya penting):
    1. EXIF        - hormati tag Orientation kamera/scanner (kalau ada).
    2. Orientasi   - koreksi kasar kelipatan 90 derajat (0/90/180/270).
    3. Deskew      - koreksi halus miring +-15 derajat.
    4. (opsional)  - potong tepi dokumen / koreksi perspektif untuk foto HP.
    5. Iluminasi   - ratakan latar (kertas krem + watermark) sebelum binarisasi.
    -> baru setelah ini ROI boleh dipotong.

Pakai sebagai modul:
    from preprocess import siapkan_halaman
    img, info = siapkan_halaman(img_bgr)

Pakai sebagai CLI (untuk mengintip hasilnya):
    python preprocess.py contoh_ijazah/ijazah_s2.jpg --debug

Dependensi:
    pip install opencv-python-headless numpy pytesseract
    sistem: tesseract-ocr + data 'osd'  (sudo apt install tesseract-ocr-osd)
"""
import argparse, os, sys
import numpy as np
import cv2


# ============================================================================
# 0. UTILITAS
# ============================================================================
def _kecilkan(gray, sisi_maks=1500):
    """Perkecil salinan untuk analisis. Analisis cepat, hasil tetap dipakai
    pada citra resolusi penuh. Skala tidak mengubah sudut maupun orientasi."""
    skala = sisi_maks / max(gray.shape[:2])
    if skala >= 1.0:
        return gray
    return cv2.resize(gray, None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA)


def _abu(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ============================================================================
# 1. EXIF  - orientasi yang direkam kamera/scanner
# ============================================================================
def terapkan_exif(path, img):
    """cv2.imread() MENGABAIKAN tag EXIF Orientation, jadi citra yang di HP
    terlihat tegak bisa termuat terbaring. Kita baca tagnya sendiri.
    Mengembalikan (citra, kode_exif) - kode 1 atau None berarti tidak ada koreksi."""
    try:
        from PIL import Image, ExifTags
    except ImportError:
        return img, None
    try:
        with Image.open(path) as im:
            kode = (im.getexif() or {}).get(274)  # 274 = Orientation
    except Exception:
        return img, None
    if not kode or kode == 1:
        return img, kode
    aksi = {
        2: lambda a: cv2.flip(a, 1),
        3: lambda a: cv2.rotate(a, cv2.ROTATE_180),
        4: lambda a: cv2.flip(a, 0),
        5: lambda a: cv2.flip(cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE), 1),
        6: lambda a: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE),
        7: lambda a: cv2.flip(cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE), 1),
        8: lambda a: cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }.get(kode)
    return (aksi(img) if aksi else img), kode


# ============================================================================
# 2. ORIENTASI KASAR (0 / 90 / 180 / 270)
# ============================================================================
def _osd_tesseract(gray):
    """Tesseract OSD: satu-satunya cara andal membedakan 0 vs 180 (butuh
    data bahasa 'osd'). Kembalikan (derajat_putar_searah_jarum_jam, keyakinan)."""
    try:
        import pytesseract
        teks = pytesseract.image_to_osd(_kecilkan(gray), config="--psm 0")
    except Exception:
        return None, 0.0
    info = {}
    for baris in teks.splitlines():
        if ":" in baris:
            k, v = baris.split(":", 1)
            info[k.strip()] = v.strip()
    try:
        # 'Rotate' = berapa derajat citra harus diputar SEARAH JARUM JAM agar tegak
        return int(info["Rotate"]) % 360, float(info.get("Orientation confidence", 0))
    except (KeyError, ValueError):
        return None, 0.0


def _proyeksi_tegak_atau_baring(gray):
    """Cadangan bila OSD tidak tersedia/ragu.

    Teks tegak = deretan baris gelap diselingi celah putih, jadi profil
    proyeksi PER-BARIS naik-turun tajam. Kalau halaman terbaring, justru
    profil PER-KOLOM yang naik-turun tajam. Ukur ketajaman dengan jumlah
    kuadrat selisih antar-elemen, lalu bandingkan.

    Hanya bisa menjawab 'perlu diputar 90 derajat atau tidak' - TIDAK bisa
    membedakan 0 vs 180 maupun 90 vs 270. Kembalikan (bool, rasio)."""
    g = _kecilkan(gray, 1200)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0
    tajam = lambda p: float(np.square(np.diff(p.astype(np.float64))).sum())
    skor_baris = tajam(bw.sum(axis=1))   # profil per-baris  -> tinggi bila tegak
    skor_kolom = tajam(bw.sum(axis=0))   # profil per-kolom  -> tinggi bila baring
    rasio = skor_kolom / max(skor_baris, 1e-9)
    return rasio > 1.0, rasio


def deteksi_orientasi(gray, pakai_osd=True, min_keyakinan=1.0):
    """Tentukan berapa derajat citra harus diputar SEARAH JARUM JAM agar tegak.
    Kembalikan (derajat, metode, keyakinan)."""
    if pakai_osd:
        derajat, keyakinan = _osd_tesseract(gray)
        if derajat is not None and keyakinan >= min_keyakinan:
            return derajat, "osd", keyakinan

    baring, rasio = _proyeksi_tegak_atau_baring(gray)
    if not baring:
        return 0, "proyeksi", rasio
    # Halaman terbaring. Arahnya (90 vs 270) tak terbaca dari proyeksi;
    # pilih 90 sebagai tebakan lalu biarkan OSD memutuskan bila tersedia.
    derajat, keyakinan = (_osd_tesseract(gray) if pakai_osd else (None, 0.0))
    if derajat in (90, 270):
        return derajat, "proyeksi+osd", keyakinan
    return 90, "proyeksi", rasio


def putar_kelipatan_90(img, derajat):
    """Putar SEARAH JARUM JAM sebesar kelipatan 90 derajat. Tanpa interpolasi,
    jadi tidak ada piksel yang rusak."""
    return {
        0: lambda a: a,
        90: lambda a: cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE),
        180: lambda a: cv2.rotate(a, cv2.ROTATE_180),
        270: lambda a: cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }[derajat % 360](img)


# ============================================================================
# 3. DESKEW - koreksi miring halus (+- 15 derajat)
# ============================================================================
def _skew_kotak_minimum(gray, batas=15.0):
    """Gabungkan huruf jadi gumpalan baris teks (dilasi memanjang), lalu ambil
    median sudut minAreaRect tiap baris. Tahan terhadap baris pencilan."""
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    inti = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    gumpal = cv2.dilate(bw, inti, iterations=2)
    kontur, _ = cv2.findContours(gumpal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sudut = []
    for k in kontur:
        if cv2.contourArea(k) < 800:          # buang bintik/noda
            continue
        (_, _), (w, h), a = cv2.minAreaRect(k)
        if w < h:                              # samakan acuan ke sisi panjang
            a += 90
        a = (a + 90) % 180 - 90                # bawa ke rentang (-90, 90]
        if abs(a) <= batas:
            sudut.append(a)
    return (float(np.median(sudut)), len(sudut)) if sudut else (None, 0)


def _skew_proyeksi(gray, kasar, jangkauan=1.5, langkah=0.1):
    """Perhalus tebakan `kasar`: cari sudut yang membuat profil proyeksi
    per-baris paling 'bergerigi' (baris teks paling rapi sejajar)."""
    g = _kecilkan(gray, 900)
    amb = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    h, w = g.shape

    def skor(a):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
        proyeksi = (r < amb).sum(axis=1).astype(np.float64)
        return np.square(np.diff(proyeksi)).sum()

    kandidat = np.arange(kasar - jangkauan, kasar + jangkauan + 1e-9, langkah)
    return float(max(kandidat, key=skor))


def estimasi_skew(gray, batas=15.0):
    """Sudut miring dalam derajat (positif = objek miring searah jarum jam)."""
    kasar, n = _skew_kotak_minimum(_kecilkan(gray, 1400), batas)
    if kasar is None or n < 3:
        return 0.0
    halus = _skew_proyeksi(gray, kasar)
    return halus if abs(halus) <= batas else 0.0


def luruskan(img, sudut):
    """Putar BERLAWANAN sebesar `sudut` untuk membatalkan miring. Kanvas
    diperbesar agar sudut halaman tidak terpotong; sisa diisi putih."""
    if abs(sudut) < 0.05:
        return img
    h, w = img.shape[:2]
    pusat = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(pusat, sudut, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    w_baru, h_baru = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += w_baru / 2.0 - pusat[0]
    M[1, 2] += h_baru / 2.0 - pusat[1]
    isi = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, M, (w_baru, h_baru),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=isi)


# ============================================================================
# 4. (OPSIONAL) POTONG TEPI DOKUMEN - untuk foto HP, bukan hasil scan
# ============================================================================
def potong_dokumen(bgr, min_luas=0.35):
    """Cari segi-empat terbesar (lembar ijazah) lalu koreksi perspektifnya.
    Berguna untuk foto HP yang miring/ada latar meja. Kalau tidak ketemu
    segi-empat yang meyakinkan, citra dikembalikan apa adanya."""
    h, w = bgr.shape[:2]
    g = _kecilkan(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 1000)
    sk = g.shape[1] / w
    tepi = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 120)
    tepi = cv2.dilate(tepi, np.ones((3, 3), np.uint8), iterations=2)
    kontur, _ = cv2.findContours(tepi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not kontur:
        return bgr, False
    k = max(kontur, key=cv2.contourArea)
    if cv2.contourArea(k) < min_luas * g.shape[0] * g.shape[1]:
        return bgr, False
    apr = cv2.approxPolyDP(k, 0.02 * cv2.arcLength(k, True), True)
    if len(apr) != 4:
        return bgr, False

    titik = apr.reshape(4, 2).astype(np.float32) / sk       # balik ke skala penuh
    jum, sel = titik.sum(1), np.diff(titik, axis=1).ravel()
    urut = np.array([titik[np.argmin(jum)], titik[np.argmin(sel)],
                     titik[np.argmax(jum)], titik[np.argmax(sel)]], np.float32)
    (kiri_atas, kanan_atas, kanan_bawah, kiri_bawah) = urut
    lebar = int(max(np.linalg.norm(kanan_bawah - kiri_bawah),
                    np.linalg.norm(kanan_atas - kiri_atas)))
    tinggi = int(max(np.linalg.norm(kanan_atas - kanan_bawah),
                     np.linalg.norm(kiri_atas - kiri_bawah)))
    if lebar < 50 or tinggi < 50:
        return bgr, False
    tujuan = np.array([[0, 0], [lebar - 1, 0], [lebar - 1, tinggi - 1], [0, tinggi - 1]], np.float32)
    M = cv2.getPerspectiveTransform(urut, tujuan)
    return cv2.warpPerspective(bgr, M, (lebar, tinggi), flags=cv2.INTER_CUBIC), True


# ============================================================================
# 5. NORMALISASI ILUMINASI - ratakan kertas krem + watermark
# ============================================================================
def normalisasi_iluminasi(gray, inti=31):
    """Perkirakan latar dengan morphological closing lalu bagi citra dengan
    latar itu. Hasilnya teks gelap di atas putih rata - Otsu jadi jauh lebih
    stabil pada kertas berwarna atau pencahayaan tidak merata."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inti, inti))
    latar = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    return cv2.divide(gray, latar, scale=255)


# ============================================================================
# PIPELINE UTAMA
# ============================================================================
def siapkan_halaman(bgr, path=None, pakai_osd=True, deskew=True,
                    crop_dokumen=False, verbose=False):
    """Tegakkan halaman ijazah. Kembalikan (citra_BGR_tegak, info_dict).

    Panggil ini SETELAH load_image() dan SEBELUM crop_roi()."""
    info = {"exif": None, "putar": 0, "metode_orientasi": None,
            "keyakinan": 0.0, "skew": 0.0, "crop_dokumen": False,
            "ukuran_awal": bgr.shape[:2][::-1]}

    # 1. EXIF
    if path:
        bgr, info["exif"] = terapkan_exif(path, bgr)

    # 4a. potong tepi dulu (kalau diminta) supaya latar meja tak mengganggu analisis
    if crop_dokumen:
        bgr, info["crop_dokumen"] = potong_dokumen(bgr)

    # 2. orientasi kasar
    derajat, metode, keyakinan = deteksi_orientasi(_abu(bgr), pakai_osd=pakai_osd)
    info.update(putar=derajat, metode_orientasi=metode, keyakinan=round(keyakinan, 2))
    if derajat:
        bgr = putar_kelipatan_90(bgr, derajat)

    # 3. deskew halus
    if deskew:
        sudut = estimasi_skew(_abu(bgr))
        info["skew"] = round(sudut, 2)
        if sudut:
            bgr = luruskan(bgr, sudut)

    info["ukuran_akhir"] = bgr.shape[:2][::-1]
    if verbose:
        print(f"[pra-proses] exif={info['exif']} putar={info['putar']}d "
              f"({info['metode_orientasi']}, conf={info['keyakinan']}) "
              f"skew={info['skew']}d "
              f"{info['ukuran_awal']} -> {info['ukuran_akhir']}", file=sys.stderr)
    return bgr, info


# ============================================================================
# CLI - untuk mengintip hasil pra-pemrosesan
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Tegakkan halaman ijazah (orientasi 90d + deskew) sebelum ROI diambil.")
    ap.add_argument("berkas", help="berkas ijazah (.pdf/.jpg/.png)")
    ap.add_argument("-o", "--keluaran", help="simpan citra tegak ke berkas ini")
    ap.add_argument("--page", type=int, default=0, help="halaman PDF (0=pertama)")
    ap.add_argument("--dpi", type=int, default=300, help="resolusi render PDF")
    ap.add_argument("--tanpa-osd", action="store_true",
                    help="jangan pakai Tesseract OSD, andalkan proyeksi saja")
    ap.add_argument("--tanpa-deskew", action="store_true", help="lewati koreksi miring halus")
    ap.add_argument("--crop-dokumen", action="store_true",
                    help="potong tepi & koreksi perspektif (untuk foto HP)")
    ap.add_argument("--debug", action="store_true",
                    help="simpan citra tegak + versi ternormalisasi ke hasil/")
    args = ap.parse_args()

    if not os.path.exists(args.berkas):
        raise SystemExit(f"Berkas tidak ditemukan: {args.berkas}")

    from deteksi_nomor_ijazah import load_image          # pemuat pdf/jpg/png yang sudah ada
    img = load_image(args.berkas, page=args.page, dpi=args.dpi)
    tegak, info = siapkan_halaman(img, path=args.berkas,
                                  pakai_osd=not args.tanpa_osd,
                                  deskew=not args.tanpa_deskew,
                                  crop_dokumen=args.crop_dokumen)

    print("=" * 60)
    print(f"Berkas          : {args.berkas}")
    print(f"Ukuran awal     : {info['ukuran_awal'][0]} x {info['ukuran_awal'][1]} px")
    print(f"EXIF Orientation: {info['exif']}")
    print(f"Putar kasar     : {info['putar']} derajat  (metode: {info['metode_orientasi']}, "
          f"keyakinan: {info['keyakinan']})")
    print(f"Skew halus      : {info['skew']} derajat")
    print(f"Crop dokumen    : {info['crop_dokumen']}")
    print(f"Ukuran akhir    : {info['ukuran_akhir'][0]} x {info['ukuran_akhir'][1]} px")
    print("=" * 60)

    if args.keluaran:
        cv2.imwrite(args.keluaran, tegak)
        print(f"disimpan: {args.keluaran}")
    if args.debug:
        os.makedirs("hasil", exist_ok=True)
        base = os.path.splitext(os.path.basename(args.berkas))[0]
        cv2.imwrite(f"hasil/pra_{base}_tegak.png", tegak)
        cv2.imwrite(f"hasil/pra_{base}_normalisasi.png",
                    normalisasi_iluminasi(_abu(tegak)))
        print(f"[debug] hasil/pra_{base}_tegak.png, hasil/pra_{base}_normalisasi.png")


if __name__ == "__main__":
    main()
