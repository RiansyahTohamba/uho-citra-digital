### **Sub-CPMK0423 - Memahami Penggunaan Histogram pada Citra Digital**  
**Deskripsi:**  
Histogram citra adalah representasi distribusi intensitas piksel dalam suatu citra. Histogram dapat digunakan untuk memahami kontras, kecerahan, dan distribusi warna dalam gambar.

**Contoh Kode (Python - OpenCV & Matplotlib):**  
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar grayscale
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

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
```

---
### **Sub-CPMK0811 - Menerapkan Transformasi Gray Level dan Histogram Equalization**  
**Deskripsi:**  
Transformasi gray level dapat meningkatkan kualitas citra dengan mengubah distribusi intensitas piksel. Histogram Equalization adalah teknik untuk meningkatkan kontras dengan menyebarkan distribusi intensitas piksel.

**Contoh Kode (Python - OpenCV):**  
```python
# Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# Tampilkan hasil
cv2.imshow("Original", image)
cv2.imshow("Equalized", equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---
