import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar dalam mode grayscale
image = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)

# Transformasi Fourier
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

# Magnitude Spectrum
magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)  # +1 untuk menghindari log(0)

# Normalisasi ke rentang 0-255 agar bisa ditampilkan dengan cv2.imshow
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
magnitude_spectrum = np.uint8(magnitude_spectrum)

# Tampilkan hasil menggunakan OpenCV
cv2.imshow("Gambar Asli", image)
cv2.imshow("Magnitude Spectrum", magnitude_spectrum)

# Tunggu input dari user lalu tutup semua jendela
cv2.waitKey(0)
cv2.destroyAllWindows()
