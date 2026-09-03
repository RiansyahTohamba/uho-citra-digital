# Setup GPU (TensorFlow + PyTorch) di WSL2

Catatan penyiapan GPU untuk `course/matmul/gpu_check.py`. Diuji pada WSL2,
NVIDIA GeForce RTX 3070 Ti Laptop (compute capability 8.6), driver 610.43.02.

## Gejala

`gpu_check.py` selalu melaporkan CPU saja, dengan pesan:

```
Could not find cuda drivers on your machine, GPU will not be used.
```

Pesan ini menyesatkan: driver-nya **tidak** bermasalah. `nvidia-smi` jalan
normal dan `libcuda.so.1` sudah ada di cache loader (`/usr/lib/wsl/lib`).

## Penyebab

Ada dua masalah terpisah:

1. **TensorFlow terpasang tanpa pustaka CUDA.** Sejak TF 2.16,
   `pip install tensorflow` hanya berisi versi CPU. Pustaka CUDA/cuDNN baru
   ikut terpasang lewat extra `[and-cuda]`.
2. **PyTorch belum terpasang sama sekali**, sehingga skrip gagal di `import
   torch` (baris 2) sebelum sampai ke pengecekan PyTorch.

Setelah `tensorflow[and-cuda]` dipasang, muncul masalah ketiga yang lebih
halus — `Cannot dlopen some GPU libraries`, dengan tepat satu pustaka gagal:

```
Could not load dynamic library 'libcusolver.so.11'
```

padahal file-nya ada di `site-packages/nvidia/cusolver/lib/`. Alasannya:
`libtensorflow_cc.so.2` memang mencantumkan direktori nvidia itu di RUNPATH-nya,
jadi semua pustaka CUDA yang **di-link langsung** ketemu. Tapi libcusolver tidak
di-link — ia dibuka lazy lewat `dlopen()` dari
`_pywrap_tensorflow_internal.so`, dan RUNPATH modul itu tidak memuat direktori
nvidia sama sekali.

## Langkah perbaikan

```bash
.venv/bin/python -m pip install "tensorflow[and-cuda]==2.21.0"
.venv/bin/python -m pip install torch torchvision
```

Lalu pasang penambal path untuk libcusolver. Pustaka dimuat lebih awal dengan
path absolut; `dlopen()` mencocokkan objek yang sudah termuat berdasarkan
SONAME, jadi `dlopen("libcusolver.so.11")` milik TensorFlow langsung dapat
handle-nya tanpa perlu mencari di path.

Penambal ini dipasang sebagai `.pth` + modul di `site-packages`, memakai import
hook sehingga preload hanya jalan saat `import tensorflow` — proses lain di
virtualenv ini (Flask, skrip OCR) tidak terbebani. Cara ini juga otomatis
berlaku di kernel Jupyter, tanpa perlu menyetel `LD_LIBRARY_PATH` di mana-mana.

File yang dibuat (keduanya di `.venv/lib/python3.12/site-packages/`):

- `_tf_cuda_path_fix.py` — modul preload, lihat docstring-nya untuk detail
- `zz_tf_cuda_path_fix.pth` — berisi satu baris: `import _tf_cuda_path_fix`

> **Penting:** `.venv` ada di `.gitignore`, jadi penambal ini **tidak**
> ikut ter-commit. Kalau virtualenv dibuat ulang atau TensorFlow di-upgrade,
> langkah ini perlu diulang. Hapus saja kedua file itu kalau rilis TensorFlow
> berikutnya sudah bisa menemukan libcusolver sendiri.

## Hasil

```
=== CEK GPU VIA TENSORFLOW ===
Status: GPU Terdeteksi (1 unit)
 - /physical_device:GPU:0

=== CEK GPU VIA PYTORCH ===
Status: GPU Terdeteksi
 - Jumlah GPU : 1
 - Nama GPU   : NVIDIA GeForce RTX 3070 Ti Laptop GPU
```

Matmul 2000x2000 sudah diverifikasi benar-benar jalan di GPU pada kedua
framework (TF: `/device:GPU:0`, PyTorch: `cuda:0`), termasuk konvolusi cuDNN.

## Catatan: jangan campur TF dan PyTorch di satu proses

TensorFlow 2.21 memakai CUDA 12 (paket `nvidia-*-cu12`), sedangkan PyTorch
2.14 memakai CUDA 13 (`site-packages/nvidia/cu13/lib`). Keduanya bisa hidup
berdampingan karena nama paket dan SONAME-nya berbeda, tapi **urutan
inisialisasi CUDA berpengaruh**:

| Urutan                                     | Hasil        |
| ------------------------------------------ | ------------ |
| TensorFlow saja                            | jalan        |
| PyTorch saja                               | jalan        |
| PyTorch pakai GPU dulu, lalu TensorFlow    | jalan        |
| TensorFlow pakai GPU dulu, lalu PyTorch    | **SIGFPE**   |

Baris terakhir crash (core dump) tepat saat PyTorch menginisialisasi context
CUDA-nya. Ini bukan soal kehabisan memori — dibatasi TF ke 1 GB pun tetap
crash.

`gpu_check.py` sendiri aman: ia hanya *mendeteksi* device dan tidak pernah
membuat context penuh. Tapi kalau nanti menulis kode yang memakai GPU dari
kedua framework dalam satu proses (atau satu kernel notebook), jalankan bagian
PyTorch lebih dulu — atau lebih baik pisahkan ke proses/notebook berbeda.

Alternatif tuntas: pasang PyTorch varian CUDA 12 agar kedua framework berbagi
satu tumpukan CUDA.

```bash
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
