import numpy as np
import cv2
"""
Tahap ABCD sebagai tahap tambahan dalam proses riset
A. hitung SNR dan VGA (before)
1. Hitung histogram gambar input (jumlah pixel untuk setiap intensitas 0-255).
2. Hitung CDF (Cumulative Distribution Function) dari histogram tersebut.
3. Normalisasi CDF supaya nilainya dari 0 ke 255.
4. Gunakan hasil normalisasi untuk mapping pixel lama(unequalized histogram) ke pixel baru(equalized histogram).
B. hitung SNR dan VGA (after)
C. hitung statistik wilcoxon
D. kesimpulan


Output gambar baru dengan pixel yang sudah di-mapping.
"""
def manual_equalizeHist(image):
    print(image.shape)
    # print(image[0].shape)
    print(image.flatten().shape)
    # rk = bins
    # Step 1: hitung histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    print(bins)
    print(bins[:-1])
    
    # print(hist.shape)

    # Step 2: hitung CDF
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # Normalisasi ke 0-255

    # Step 3: mapping pixel berdasarkan CDF
    equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)

    # Step 4: reshape jadi ukuran gambar semula
    return equalized.reshape(image.shape).astype(np.uint8)

image = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)
equalized = manual_equalizeHist(image)

# print(equalized)
# print(equalized.shape)