import cv2
import numpy as np
import matplotlib.pyplot as plt

# Transformasi Fourier digunakan untuk menganalisis komponen frekuensi dalam citra, yang berguna dalam filtering domain frekuensi.
image = cv2.imread('sample.png')

# **Contoh Kode (Python - NumPy & OpenCV):**  
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

# Magnitude Spectrum
magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

# Tampilkan hasil
plt.figure()
plt.title("Magnitude Spectrum")
plt.imshow(magnitude_spectrum, cmap='gray')
plt.show()