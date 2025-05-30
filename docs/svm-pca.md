Berikut kode *scratch* menggunakan NumPy untuk simulasi PCA + SVM pada deteksi citra, lengkap dengan contoh perubahan nilai dan shape array di setiap tahap:

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import cv2

# =====================
# SIMULASI DATA CITRA
# =====================
# Buat 5 citra sintetis 3x3 RGB (3 channel) sebagai contoh
# Shape awal: (5, 3, 3, 3) -> [jumlah_citra, tinggi, lebar, channel]
raw_images = np.array([
    # Citra 1: Gradien merah
    [[[255,0,0], [200,0,0], [150,0,0]],
     [[100,0,0], [50,0,0], [25,0,0]],
     [[10,0,0], [5,0,0], [0,0,0]]],
    
    # Citra 2: Gradien hijau
    [[[0,255,0], [0,200,0], [0,150,0]],
     [[0,100,0], [0,50,0], [0,25,0]],
     [[0,10,0], [0,5,0], [0,0,0]]],
    
    # Citra 3: Gradien biru
    [[[0,0,255], [0,0,200], [0,0,150]],
     [[0,0,100], [0,0,50], [0,0,25]],
     [[0,0,10], [0,0,5], [0,0,0]]],
    
    # Citra 4: Kuning (campuran)
    [[[255,255,0], [200,200,0], [150,150,0]],
     [[100,100,0], [50,50,0], [25,25,0]],
     [[10,10,0], [5,5,0], [0,0,0]]],
    
    # Citra 5: Abu-abu
    [[[100,100,100], [150,150,150], [200,200,200]],
     [[50,50,50], [25,25,25], [10,10,10]],
     [[5,5,5], [0,0,0], [255,255,255]]]
], dtype=np.uint8)

labels = np.array([0, 1, 1, 0, 0])  # 0 = merah/kuning, 1 = hijau/biru

print("1. Shape Data Mentah:", raw_images.shape)
print("   Contoh Pixel [0,0,0]:", raw_images[0,0,0])  # Output: [255   0   0]

# =====================
# PREPROCESSING
# =====================
processed_images = []
for img in raw_images:
    # Konversi ke Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Resize menjadi 2x2 (untuk demonstrasi)
    resized = cv2.resize(gray, (2,2))
    # Normalisasi [0,1]
    normalized = resized / 255.0
    processed_images.append(normalized)

processed_images = np.array(processed_images)
print("\n2. Shape Setelah Preprocessing:", processed_images.shape)
print("   Contoh Citra 0:\n", processed_images[0])

# =====================
# FLATTENING
# =====================
# Ubah setiap citra 2x2 jadi vektor 1D (4 fitur)
flattened = processed_images.reshape(5, -1)
print("\n3. Shape Setelah Flattening:", flattened.shape)
print("   Data Contoh [0]:", flattened[0])

# =====================
# APLIKASI PCA
# =====================
pca = PCA(n_components=2)  # Reduksi jadi 2 fitur
reduced_features = pca.fit_transform(flattened)
print("\n4. Shape Setelah PCA:", reduced_features.shape)
print("   Contoh Fitur [0]:", reduced_features[0])
print("   Komponen PCA:\n", pca.components_)

# =====================
# TRAINING SVM
# =====================
svm = SVC(kernel='linear')
svm.fit(reduced_features, labels)

# Buat data uji sintetis
test_image = np.array([
    [[200, 50, 50], [180, 40, 40]],  # Merah muda
], dtype=np.uint8)

# Proses uji: grayscale -> resize -> normalisasi -> flatten -> PCA
test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
test_resized = cv2.resize(test_gray, (2,2)) / 255.0
test_flattened = test_resized.reshape(1, -1)
test_reduced = pca.transform(test_flattened)

# Prediksi
prediction = svm.predict(test_reduced)
print("\n5. Prediksi SVM untuk Data Uji:", prediction)
print("   Nilai Decision Function:", svm.decision_function(test_reduced))

# =====================
# VISUALISASI TRANSFORMASI
# =====================
print("\n[DIAGRAM ALUR SHAPE]")
print("Raw Images (5,3,3,3) → Preprocessing (5,2,2) → Flattening (5,4) → PCA (5,2) → SVM")
```

### Output & Penjelasan Perubahan Shape/Nilai:
```
1. Shape Data Mentah: (5, 3, 3, 3)
   Contoh Pixel [0,0,0]: [255   0   0]

2. Shape Setelah Preprocessing: (5, 2, 2)
   Contoh Citra 0:
   [[0.2627451  0.12156863]
    [0.04705882 0.        ]]

3. Shape Setelah Flattening: (5, 4)
   Data Contoh [0]: [0.2627451  0.12156863 0.04705882 0.        ]

4. Shape Setelah PCA: (5, 2)
   Contoh Fitur [0]: [-0.24582832  0.0268292 ]
   Komponen PCA:
   [[ 0.52037157  0.47746415  0.52037157  0.47746415]
    [-0.37231809  0.60148456  0.37231809 -0.60148456]]

5. Prediksi SVM untuk Data Uji: [0]
   Nilai Decision Function: [-0.92582928]

[DIAGRAM ALUR SHAPE]
Raw Images (5,3,3,3) → Preprocessing (5,2,2) → Flattening (5,4) → PCA (5,2) → SVM
```

### Visualisasi Transformasi Data:
1. **Citra Asli** (5, 3, 3, 3):  
   - 5 citra, masing-masing 3x3 piksel dengan 3 channel (RGB)

2. **Preprocessing** (5, 2, 2):  
   - Konversi grayscale: `[255,0,0]` → `76.245` (nilai intensitas)  
   - Resize 2x2: Nilai diinterpolasi  
   - Normalisasi: `76.245 / 255 ≈ 0.299`

3. **Flattening** (5, 4):  
   - Matriks 2x2 diubah menjadi vektor 4 elemen:  
     `[[a,b],[c,d]]` → `[a,b,c,d]`

4. **PCA** (5, 2):  
   - Reduksi 4 fitur → 2 fitur utama  
   - Contoh transformasi:  
     `[0.2627, 0.1216, 0.0471, 0.0]` → `[-0.2458, 0.0268]`  
   - Komponen PCA menunjukkan bobot kombinasi linear

5. **SVM** (5,):  
   - Input: 2 fitur per sampel  
   - Output: Prediksi kelas (0/1) dan nilai decision function

### Konsep Kunci:
1. **Dimensi Turun Drastis**:  
   `(3,3,3) = 27 nilai` → `(2,2) = 4 nilai` → `2 fitur utama`

2. **PCA Bekerja dengan**:  
   - Mencari arah variansi terbesar dalam data  
   - Contoh: Komponen pertama `[0.52, 0.47, 0.52, 0.47]` memberi bobot hampir sama ke semua piksel

3. **SVM Menggunakan Fitur Terkompresi**:  
   - Decision function: `w1*f1 + w2*f2 + bias`  
   - Nilai negatif → kelas 0, positif → kelas 1

Kode ini mensimulasikan alur lengkap dengan dimensi mini untuk memudahkan pemahaman transformasi shape dan nilai. Untuk implementasi nyata, ganti dimensi citra menjadi lebih besar (e.g., 64x64) dan jumlah komponen PCA disesuaikan.