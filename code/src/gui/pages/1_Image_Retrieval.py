import sys
import sqlite3
import cv2
import numpy as np
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

from features.feature_extraction import *
from retrieval.search import *
from util.retriever import Retriever

@st.cache_resource
def get_db():
    return sqlite3.connect(
        ROOT/"database"/"multimedia.db",
        check_same_thread=False
    )


retriever = Retriever(get_db()) 

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
        # results = image_similarity_search(db, query_img, top_k)
        # results = fetch_image_from_db(db, results)

        st.subheader("Results")
        cols = st.columns(len(results))

        st.write(str(results))

        for col, (id, dist, img) in zip(cols, results):
            col.image(
                img,
                caption=f"media_id={id}\nSimilarity Score={dist:.3f}",
                width='stretch',
                channels='BGR')
                
