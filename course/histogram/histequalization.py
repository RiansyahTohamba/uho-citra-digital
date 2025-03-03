
import cv2
# Histogram Equalization
image = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)

equalized_image = cv2.equalizeHist(image)

# Tampilkan hasil
cv2.imshow("Original", image)
cv2.imshow("Equalized", equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()