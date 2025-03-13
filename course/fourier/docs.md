Sinyal dan citra sebenarnya memiliki hubungan yang erat dalam domain frekuensi karena **citra adalah sinyal dalam bentuk dua dimensi**. Untuk memahami mengapa ada sinyal frekuensi pada citra, kita perlu melihat keterkaitan antara sinyal 1D dan citra 2D:

### 1. **Sinyal sebagai Fungsi dalam Domain Waktu dan Ruang**
   - Sinyal 1D: Biasanya kita menganggap sinyal sebagai fungsi dari **waktu** (misalnya, sinyal suara).
   - Citra 2D: Citra adalah sinyal dalam **ruang**, dengan domain koordinat **(x, y)** yang merepresentasikan lokasi piksel.

### 2. **Frekuensi dalam Sinyal dan Citra**
   - Dalam sinyal 1D, **frekuensi tinggi** berarti perubahan cepat dalam waktu (seperti suara nada tinggi), dan **frekuensi rendah** berarti perubahan lambat.
   - Dalam citra 2D, **frekuensi tinggi** berarti ada perubahan intensitas yang tajam dalam ruang (seperti tepi atau tekstur detail), sedangkan **frekuensi rendah** berarti perubahan intensitas yang halus (seperti daerah dengan warna atau terang yang seragam).

### 3. **Transformasi Fourier dalam Citra**
   - Transformasi Fourier dalam sinyal 1D mengubah data dari domain waktu ke domain frekuensi.
   - Dalam citra 2D, kita menggunakan **Transformasi Fourier 2D** (`np.fft.fft2`) untuk mengubah data dari domain spasial (koordinat piksel) ke domain frekuensi.

### 4. **Interpretasi Domain Frekuensi dalam Citra**
   - Frekuensi rendah dalam citra biasanya berpusat di tengah domain Fourier dan mewakili informasi global seperti pencahayaan atau warna keseluruhan.
   - Frekuensi tinggi berada di tepi domain Fourier dan menangkap detail seperti tepi dan tekstur.
   - Jika kita menyaring frekuensi tinggi, citra menjadi buram (karena detail tajam dihilangkan).
   - Jika kita menyaring frekuensi rendah, citra hanya menampilkan tepi (mirip dengan deteksi tepi).

### **Kesimpulan**
Citra memiliki sinyal frekuensi karena perubahan intensitas piksel dalam ruang dapat direpresentasikan sebagai kombinasi gelombang sinusoidal dengan berbagai frekuensi. Transformasi Fourier digunakan dalam citra untuk berbagai aplikasi, seperti kompresi (JPEG), deteksi tepi, dan pemrosesan filter spasial.

Setelah memahami ini, kita bisa lanjut ke penerapan **FFT 2D pada citra** untuk melihat bagaimana frekuensi muncul dalam gambar.