
import cv2
# Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# Tampilkan hasil
cv2.imshow("Original", image)
cv2.imshow("Equalized", equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()