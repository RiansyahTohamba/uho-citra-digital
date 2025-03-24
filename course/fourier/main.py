import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Baca gambar dalam mode grayscale
image_path = 'sample.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ukuran asli gambar dalam byte
original_size = os.path.getsize(image_path)

# Ukuran asli gambar dalam piksel
original_shape = image.shape

# Transformasi Fourier
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

# Magnitude Spectrum
magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)  # +1 untuk menghindari log(0)

# Normalisasi agar bisa disimpan sebagai gambar
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
magnitude_spectrum = np.uint8(magnitude_spectrum)

# Simpan gambar hasil transformasi Fourier sementara
fourier_image_path = 'fourier_output.png'
cv2.imwrite(fourier_image_path, magnitude_spectrum)

# Ukuran setelah transformasi dalam byte
transformed_size = os.path.getsize(fourier_image_path)

# Ukuran setelah transformasi dalam piksel
transformed_shape = magnitude_spectrum.shape

# Tampilkan gambar asli dan hasil transformasi
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image, cmap='gray')
axes[0].set_title(f'Gambar Asli (Size: {original_shape}, {original_size} bytes)')
axes[0].axis('off')

axes[1].imshow(magnitude_spectrum, cmap='gray')
axes[1].set_title(f'Magnitude Spectrum (Size: {transformed_shape}, {transformed_size} bytes)')
axes[1].axis('off')

plt.show()
