import cv2
import numpy as np

# Membaca gambar
citra = cv2.imread("sample.png")

# Membaca channel warna BGR dan menyimpannya ke dalam variabel terpisah
b = citra[:, :, 0]
g = citra[:, :, 1]
r = citra[:, :, 2]

# Menyimpan jumlah baris dan jumlah kolom dari citra
jum_baris = len(citra)
jum_kolom = len(citra[0])

# Menyiapkan citra baru dengan nilai 0
citra_gray = np.zeros((jum_baris, jum_kolom))

# Menghitung nilai pixel grayscale
for i in range(jum_baris):
    for j in range(jum_kolom):
        citra_gray[i, j] = round(0.299 * r[i, j] + 0.587 * g[i, j] + 0.114 * b[i, j])

# Konversi citra grayscale ke tipe data uint8
citra_gray = citra_gray.astype(np.uint8)

# Menampilkan citra grayscale
cv2.imshow("hmbtn gray", citra_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Menampilkan nilai matriks citra grayscale
print(citra_gray)
