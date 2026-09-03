# Histogram citra adalah representasi distribusi intensitas piksel dalam suatu citra. 
# Histogram dapat digunakan untuk memahami kontras, kecerahan, dan distribusi warna dalam gambar.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar grayscale
image = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)
print(image.shape)
cv2.imshow('gray',image)

# Hitung histogram
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Tampilkan histogram
plt.figure()
plt.title("Histogram Grayscale")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Frekuensi")
plt.plot(histogram)
plt.xlim([0, 256])
plt.show()