
import cv2
# mengubah kontras pada setiap pixel. 
# kontras berarti pixel yang gelap bertambah gelap, sedang pixel terang menjadi lebih terang

# Histogram Equalization
image = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)

equalized_image = cv2.equalizeHist(image)

# Tampilkan hasil
cv2.imshow("Original", image)
cv2.imshow("Equalized", equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()