# ### **Sub-CPMK0852 - Penggunaan Transformasi Wavelet dalam Pemrosesan Citra**  
# **Deskripsi:**  
# Transformasi Wavelet digunakan untuk ekstraksi fitur atau kompresi citra dengan dekomposisi multi-resolusi.
import pywt
import pywt.data
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lakukan transformasi wavelet
coeffs2 = pywt.dwt2(image, 'haar')
cA, (cH, cV, cD) = coeffs2  # Approximation, Horizontal, Vertical, Diagonal

# Tampilkan hasil
plt.figure(figsize=(8,8))
plt.subplot(2,2,1), plt.imshow(cA, cmap='gray'), plt.title('Approximation')
plt.subplot(2,2,2), plt.imshow(cH, cmap='gray'), plt.title('Horizontal')
plt.subplot(2,2,3), plt.imshow(cV, cmap='gray'), plt.title('Vertical')
plt.subplot(2,2,4), plt.imshow(cD, cmap='gray'), plt.title('Diagonal')
plt.show()
