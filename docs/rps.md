Berikut adalah usulan **CPMK (Capaian Pembelajaran Mata Kuliah)** dan **deskripsi singkat** untuk mata kuliah **Pengantar Pengolahan Citra Digital** berdasarkan cakupan yang telah diberikan:  

---

### 📌 **1. Capaian Pembelajaran Mata Kuliah (CPMK):**  
Setelah mengikuti mata kuliah ini, mahasiswa diharapkan mampu:  

1. **Memahami** konsep dasar pengolahan citra digital dan teori warna dalam representasi citra.  
2. **Mengimplementasikan** teknik dasar pengolahan citra menggunakan pustaka OpenCV dan Pyplot.  
3. **Menerapkan** metode peningkatan kualitas citra (image enhancement) pada domain spasial, termasuk transformasi gray level, histogram equalization, korelasi, konvolusi, smoothing, dan sharpening filters.  
4. **Menganalisis** citra digital menggunakan teknik transformasi frekuensi, khususnya transformasi Fourier.  
5. **Memecahkan masalah** degradasi citra dengan teknik restorasi citra digital.  
6. **Menggunakan** transformasi wavelet dalam pemrosesan citra digital untuk ekstraksi fitur atau kompresi citra.  

---

### 📌 **2. Deskripsi Singkat Mata Kuliah:**  
Mata kuliah **Pengantar Pengolahan Citra Digital** ini memberikan pemahaman dasar mengenai teknik dan konsep pengolahan citra, mulai dari teori warna hingga teknik peningkatan kualitas citra (image enhancement) baik dalam domain spasial maupun frekuensi. Mahasiswa akan mempelajari penggunaan pustaka OpenCV dan Pyplot dalam pemrograman pengolahan citra serta memperdalam pemahaman mengenai transformasi Fourier dan wavelet. Mata kuliah ini juga mencakup metode untuk memperbaiki citra yang mengalami degradasi, memberikan keterampilan praktis yang relevan dalam berbagai aplikasi teknologi dan penelitian.  


- Sub-CPMK0421	Memahami konsep dasar pengolahan citra digital.
- Sub-CPMK0422	Memahami teori warna dalam representasi citra digital.
- Sub-CPMK0423	Memahami penggunaan histogram pada citra digital.
- Sub-CPMK0811	Menerapkan metode peningkatan kualitas citra dengan transformasi gray level dan histogram equalization.
- Sub-CPMK0821	Menerapkan metode peningkatan kualitas citra dengan korelasi, konvolusi, smoothing, dan sharpening filters.
- Sub-CPMK0841	Menganalisis citra digital menggunakan teknik transformasi frekuensi, khususnya transformasi Fourier.
- Sub-CPMK0851	Memecahkan masalah degradasi citra dengan teknik restorasi citra digital.
- Sub-CPMK0852	Menggunakan transformasi wavelet dalam pemrosesan citra digital untuk ekstraksi fitur atau kompresi citra.
- Sub-CPMK0831	Mengimplementasikan teknik dasar pengolahan citra untuk menyelesaikan permasalahan di dunia nyata.


Berikut adalah deskripsi singkat dan contoh kode untuk setiap sub-CPMK:  

---




### **Sub-CPMK0821 - Menerapkan Filtering (Korelasi, Konvolusi, Smoothing, Sharpening)**  
**Deskripsi:**  
Filtering digunakan untuk meningkatkan kualitas citra, seperti menghaluskan (smoothing) atau menajamkan (sharpening) detail dalam citra menggunakan teknik konvolusi.

**Contoh Kode (Python - OpenCV):**  
```python
# Kernel untuk smoothing (blur)
blur_kernel = np.ones((5, 5), np.float32) / 25
smoothed = cv2.filter2D(image, -1, blur_kernel)

# Kernel untuk sharpening
sharpen_kernel = np.array([[0, -1, 0], 
                           [-1, 5, -1], 
                           [0, -1, 0]])
sharpened = cv2.filter2D(image, -1, sharpen_kernel)

# Tampilkan hasil
cv2.imshow("Smoothed", smoothed)
cv2.imshow("Sharpened", sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

### **Sub-CPMK0841 - Analisis Citra dengan Transformasi Fourier**  
**Deskripsi:**  
Transformasi Fourier digunakan untuk menganalisis komponen frekuensi dalam citra, yang berguna dalam filtering domain frekuensi.

**Contoh Kode (Python - NumPy & OpenCV):**  
```python
# Transformasi Fourier
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

# Magnitude Spectrum
magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

# Tampilkan hasil
plt.figure()
plt.title("Magnitude Spectrum")
plt.imshow(magnitude_spectrum, cmap='gray')
plt.show()
```

---

### **Sub-CPMK0851 - Restorasi Citra Digital dari Degradasi**  
**Deskripsi:**  
Restorasi citra bertujuan untuk memperbaiki citra yang mengalami degradasi, misalnya akibat noise atau blur.

**Contoh Kode (Python - Wiener Filter dengan SciPy):**  
```python
from scipy.signal import wiener

# Restorasi menggunakan Wiener Filter
restored_image = wiener(image, (5, 5))

# Tampilkan hasil
plt.subplot(1,2,1)
plt.title("Degraded Image")
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.title("Restored Image")
plt.imshow(restored_image, cmap='gray')

plt.show()
```

---

### **Sub-CPMK0852 - Penggunaan Transformasi Wavelet dalam Pemrosesan Citra**  
**Deskripsi:**  
Transformasi Wavelet digunakan untuk ekstraksi fitur atau kompresi citra dengan dekomposisi multi-resolusi.

**Contoh Kode (Python - PyWavelets):**  
```python
import pywt
import pywt.data

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
```

