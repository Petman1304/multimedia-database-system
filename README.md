# Multimedia Database System

Sistem *Multimedia Database* untuk pencarian konten berbasis fitur visual citra dan video. Sistem ini dirancang untuk mendukung penyimpanan, ekstraksi fitur, dan pencarian media (image & video) menggunakan pendekatan *Content‑Based Retrieval* (CBR).

---

## 🧠 Deskripsi

Proyek ini bertujuan membangun sistem basis data multimedia yang mampu:
- Menyimpan data citra dan video beserta metadata dan vektor fitur.
- Mengimplementasikan pencarian berbasis konten (*content‑based retrieval*).
- Menggunakan metode similarity search seperti **Euclidean**, **Cosine**, dan **KNN**.
- Mendukung *query by example* dan pencarian metadata.


---

## 🧱 Fitur Utama

- **Database multimedia**: menyimpan path media, metadata, dan vektor fitur.
- **Ekstraksi fitur citra & video**:
  - Warna (CIE Lab*), entropi, deteksi tepi (Roberts, Sobel), tekstur (Gabor).
  - Untuk video: ekstraksi keyframe → fitur citra → agregasi vektor.
- **Pencarian konten visual**:
  - Similarity search (Euclidean Distance, Cosine Similarity, KNN).
  - *Query by example* menggunakan file citra/video sebagai kueri.
- **Antarmuka visual**: aplikasi web sederhana untuk upload kueri dan melihat hasil.

---

## 📂 Struktur Proyek



```
.
├── code/                      # Kode inti sistem (ekstraksi fitur, search, dll)
├── database/                  # Sampel dan skema database
├── requirements.txt           # Daftar dependensi Python
├── packages.txt               # Paket tambahan (untuk deployment streamlit community cloud)
├── .gitignore
└── README.md
```

---

## ⚙️ Teknologi

| Komponen                   | Teknologi / Library                           |
|---------------------------|-----------------------------------------------|
| Bahasa Pemrograman        | Python                                        |
| Ekstraksi Fitur Citra     | OpenCV, scikit‑image                           |
| Keyframe Video            | PyAv                                          |
| Similarity Search         | scikit‑learn                                  |
| GUI / Presentation Layer  | Streamlit                                     |
| Database                  | SQLite3                                       |

---

## 🚀 Instalasi & Setup

1. **Clone repo:**
   ```bash
   git clone https://github.com/Petman1304/multimedia-database-system.git
   cd multimedia-database-system
   ```

2. **Virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Streamlit UI:**
   ```bash
   cd code/gui
   streamlit run Home.py  
   ```

5. **Database:**
   - Pastikan file SQLite sudah ter‑create sebelum digunakan.
   - Jalankan skrip populate_db jika belum.

---

## 🧪 Pengujian & Evaluasi

Pengujian dilakukan pada dataset citra dan video untuk mengukur *accuracy* dan *mean average precision* (mAP) sistem retrieval.

**Image Retrieval:**
- Metode pencarian (Euclidean, Cosine, KNN) menghasilkan akurasi dan mAP yang sama → metode tidak berpengaruh signifikan.
- Nilai *top_k* yang lebih tinggi menaikkan akurasi tapi mAP cenderung stabil.

**Video Retrieval:**
- Performa lebih rendah dibanding citra.
- Peningkatan *top_k* meningkatkan akurasi dengan mAP yang relatif stagnan.

---

## 🛠️ Cara Menggunakan

1. **Upload media kueri (image/video).**
2. Sistem mengekstraksi fitur media kueri.
3. Hitung similarity terhadap database.
4. Hasil ditampilkan berdasarkan ranking kemiripan.

---

