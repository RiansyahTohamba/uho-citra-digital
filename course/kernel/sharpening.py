import cv2
import numpy as np

# Baca gambar grayscale
image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Definisi kernel sharpening
sharpen_kernel = np.array([[0, -1, 0], 
                           [-1, 5, -1], 
                           [0, -1, 0]])

# Aplikasikan konvolusi dengan kernel
sharpened_image = cv2.filter2D(image, -1, sharpen_kernel)

# Tampilkan hasil
cv2.imshow("Original", image)
cv2.imshow("Sharpened", sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
