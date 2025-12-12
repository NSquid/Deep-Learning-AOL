# Potato Leaf Disease Classification Using Deep Learning

Proyek ini bertujuan untuk mengembangkan sistem otomatis untuk mengklasifikasikan penyakit daun kentang menggunakan *Deep Learning*. Studi ini membandingkan kinerja dua pendekatan arsitektur: **Custom CNN** dan **Transfer Learning (ResNet50)**.

Sistem ini telah dideploy menjadi aplikasi web berbasis Streamlit untuk prediksi *real-time*.

## Tim Pengembang 
  * **Mikhael Henokh Santoso** (2702239750)
  * **Frederick Krisna Suryopranoto** (2702272382)
  * **Dustin Manuel** (2702296016)

## Tujuan & Latar Belakang

Petani sering menghadapi kesulitan dalam mendeteksi penyakit tanaman secara dini karena metode inspeksi manual yang lambat dan mahal. Kentang sangat rentan terhadap penyakit seperti *Early Blight* dan *Late Blight*.

Tujuan proyek ini adalah:

1.  Mengembangkan sistem otomatis identifikasi penyakit daun kentang.
2.  Membandingkan performa Custom CNN vs ResNet50.
3.  Menyediakan aplikasi web yang mudah digunakan oleh petani.

## Dataset

Dataset yang digunakan terdiri dari **4.072 gambar** daun kentang yang terbagi menjadi 3 kelas:

  * **Early Blight:** 1.628 gambar.
  * **Late Blight:** 1.424 gambar.
  * **Healthy:** 1.020 gambar.

Pembagian data (Data Splitting):

  * Training: 3.251 gambar
  * Validation: 416 gambar
  * Test: 405 gambar

## Metodologi & Arsitektur Model

### 1\. Data Preprocessing

  * Resize gambar ke 224x224 piksel.
  * Augmentasi: Random horizontal flips, rotasi, penyesuaian warna, dan normalisasi.

### 2\. Model

  * **Custom CNN:** Arsitektur ringan dengan 4 blok konvolusional sekuensial.
  * **ResNet50 (Transfer Learning):** Menggunakan bobot pre-trained ImageNet dengan 47 layer awal dibekukan (*frozen*) dan layer klasifikasi dimodifikasi.

## Hasil Evaluasi (Performance)

Hasil eksperimen menunjukkan bahwa **Custom CNN** mengungguli ResNet50 dalam akurasi dan efisiensi penyimpanan untuk tugas spesifik ini[cite: 158].

| Metrik | Custom CNN | ResNet50 |
| :--- | :--- | :--- |
| **Akurasi** | **99.26%** | 98.77% |
| **Precision (Avg)** | 99.27% | 98.78% |
| **Recall (Avg)** | 99.26% | 98.77% |
| **F1-Score (Avg)** | 99.26% | 98.77% |
| **Ukuran Model** | **26.49 MB** | 93.68 MB |

**Kesimpulan Hasil:**
Custom CNN lebih efisien dan akurat dibandingkan ResNet50 untuk dataset ini, meskipun ResNet50 memiliki sedikit keunggulan dalam kecepatan inferensi. Kesalahan klasifikasi utama terjadi antara *Early Blight* dan *Late Blight* karena kemiripan visual, namun sangat jarang salah memprediksi daun sehat.

## Deployment & Link Penting

Aplikasi dideploy menggunakan **Streamlit Cloud** yang memungkinkan pengguna mengunggah gambar dan mendapatkan prediksi penyakit secara *real-time*.

  * **GitHub Repository:** [https://github.com/NSquid/Deep-Learning-AOL](https://github.com/NSquid/Deep-Learning-AOL)
  * **Aplikasi Web (Live Demo):** [https://potatoes-disease-classifications.streamlit.app/](https://potatoes-disease-classifications.streamlit.app/) 
  * **Slide Presentasi:** [Canva Link](https://www.canva.com/design/DAG7GJiBW3o/9eeT19_0ew4cT5dRQFP-5Q/edit?utm_content=DAG7GJiBW3o&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
  * **Video Demo:** [Google Drive Link](https://drive.google.com/file/d/1e5LJiWHrVBlMoY6yfqvmAjIWVcsTzjtO/view?usp=sharing)

