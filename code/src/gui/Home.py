import sys
import sqlite3
import cv2
import numpy as np
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from features.feature_extraction import *
from retrieval.search import *

st.set_page_config(
    page_title="Home"
)



st.title("Multimedia Retrieval System")

st.markdown(
"""
Sistem ini merupakan **aplikasi pencarian multimedia** yang mendukung **image retrieval** dan **video retrieval**
menggunakan pendekatan **content-based retrieval**.

Pengguna dapat melakukan pencarian berdasarkan:
- **Kemiripan visual (feature-based similarity)**
- **Metadata filtering**
- **Hybrid search (visual + metadata)**

Sistem ini dirancang untuk mendukung eksperimen dan evaluasi performa pada domain
**multimedia database dan information retrieval**.
"""
)

st.header("Main Features")

st.markdown(
"""
### Image Retrieval
- Ekstraksi fitur visual berbasis vektor
- Pencarian menggunakan:
  - Euclidean Distance
  - Cosine Similarity
  - K-Nearest Neighbors (KNN)
- Filter metadata:
  - Resolusi (width & height)
  - Ukuran file
  - Format gambar

### Video Retrieval
- Pencarian berbasis **keyframe**
- Aggregasi similarity keyframe → video
- Dukungan video format AVI (internal) dengan konversi MP4 untuk tampilan
- Ranking video berdasarkan similarity score
"""
)

st.header("System Workflow")

st.markdown(
"""
1. **Data Storage**
   - Media disimpan dalam database
   - Feature vektor disimpan terpisah untuk efisiensi pencarian

2. **Query Processing**
   - Query image / video diproses menjadi feature vector
   - Metadata filter diterapkan (opsional)

3. **Similarity Search**
   - Perhitungan jarak / similarity
   - KNN digunakan untuk pencarian tetangga terdekat

4. **Result Aggregation**
   - Pooling similarity (min / max / average)
   - Ranking hasil akhir

5. **Visualization**
   - Image dan video ditampilkan pada antarmuka Streamlit
"""
)

st.header("Technical Notes")

st.markdown(
"""
- Backend menggunakan **SQLite**
- Feature extraction dilakukan secara offline
- Streamlit digunakan sebagai antarmuka interaktif
- FFmpeg digunakan untuk konversi video saat visualisasi
"""
)

st.markdown("---")
st.caption("Multimedia Database System | Content-Based Retrieval")







