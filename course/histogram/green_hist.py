import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar grayscale
image = cv2.imread('sample.png')
only_green = image[:,:,1]

print(image.shape)

cv2.imshow('all',image)
cv2.imshow('green',only_green)
# cv2.normalize()

# # Hitung histogram
histogram = cv2.calcHist([only_green], [0], None, [256], [0, 256])

# # Tampilkan histogram
plt.figure()
plt.title("Histogram green")
plt.xlabel("Intensitas Piksel")
plt.ylabel("Frekuensi")
plt.plot(histogram)
plt.xlim([0, 256])
plt.show()
# cv2.waitKey()