
# Transformasi gray level dapat meningkatkan kualitas citra dengan mengubah distribusi intensitas piksel. 
# Histogram Equalization adalah teknik untuk meningkatkan kontras dengan menyebarkan distribusi intensitas piksel.
# mengubah kontras pada setiap pixel. 
# kontras berarti pixel yang gelap bertambah gelap, sedang pixel terang menjadi lebih terang

import cv2
image = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(image)

# Tampilkan hasil
cv2.imshow("Original", image)
cv2.imshow("Equalized", equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()