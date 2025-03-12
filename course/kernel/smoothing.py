import cv2
import numpy as np
# Filtering digunakan untuk meningkatkan kualitas citra, seperti menghaluskan (smoothing) atau menajamkan (sharpening) detail dalam citra menggunakan teknik konvolusi.

image = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# Kernel untuk smoothing (blur)
blur_kernel = np.ones((5, 5), np.float32) / 25
smoothed = cv2.filter2D(image, -1, blur_kernel)


# Tampilkan hasil
cv2.imshow("Smoothed", smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows()