import sys
import sqlite3
import cv2
import numpy as np
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from features.feature_extraction import *
from retrieval.search import *
from util.retriever import Retriever


@st.cache_resource
def get_db():
    return sqlite3.connect(
        ROOT/"database"/"multimedia.db",
        check_same_thread=False
    )

base_dir = os.path.join(Path(__file__).resolve().parents[4])

retriever = Retriever(get_db(), base_dir) 

# UI

st.title("Image Retrieval System")

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "bmp"]
)

top_k = st.slider("Top K", 1, 10, 5)

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    query_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Query")
    st.image(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB), width=300)

    if st.button("Search"):
        results = retriever.image_similarity_search(query_img, top_k)
        results = retriever.fetch_image_from_db(results)

        st.subheader("Results")
        n_cols = 3 

        for i in range(0, len(results), n_cols):
            cols = st.columns(n_cols)
            for col, (id, dist, img) in zip(cols, results[i:i+n_cols]):
                col.image(
                    img,
                    caption=f"Media ID={id}\nSimilarity Score={dist:.3f}",
                    channels="BGR",
                    use_container_width=True
                )
                
