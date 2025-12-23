import sys
import sqlite3
import cv2
import numpy as np
import streamlit as st
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from retrieval.search import *
from util.feature_extraction import *
from util.retriever import Retriever


@st.cache_resource
def get_db():
    return sqlite3.connect(
        ROOT/"database"/"multimedia.db",
        check_same_thread=False
    )

base_dir = os.path.join(Path(__file__).resolve().parents[4])

retriever = Retriever(get_db(), base_dir) 

def video_caption(fn, size, dur, fps, ext):
    caption =f"""
    File name= {fn}\n
    Size= {size/(1024)} KB\n
    Duration= {dur} px\n
    FPS= {fps} px\n
    File extension= {ext}
    """
    return caption

# UI

st.title("Video Retrieval System")

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "bmp"]
)

# Search by metadata
with st.popover("Advanced Query"):
    search_method = st.segmented_control(
        "Search Method", 
        ["Euclidean Distance", "Cosine Similarity", "KNN"],
        selection_mode="single",
        default="Euclidean Distance"
    )
    if search_method == None:
        st.warning("Please select a method")

    max_size = st.slider("Video Size (KB)", 0, 4096, 1024)
    min_fps, max_fps = st.slider("Video FPS", 0, 75, (0, 60))
    min_dur, max_dur = st.slider("Video Duration", 0, 300, (0, 60))
    ext = st.multiselect(
        "Video Extension",
        ["avi", "mp4", "mkv"],
        default=["avi", "mp4", "mkv"])

    top_k = st.slider("Top K", 1, 10, 5)

query= None
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    query = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Query")
    st.image(cv2.cvtColor(query, cv2.COLOR_BGR2RGB), width=300)

if st.button("Search"):
    start_time = time.perf_counter()

    filtered_vid = retriever.video_metadata_filter(max_size*1024, min_fps, max_fps, min_dur, max_dur, ext)
    filtered_vid = [row[0] for row in filtered_vid]

    if query is None:
        results = [(id, 0.) for id in filtered_vid]
        results = results[:top_k]
    else:
        results = retriever.video_similarity_search(query, search_method=search_method, top_k=top_k)
        results = [(id, dist) for id, dist in results if id in filtered_vid]
        results = results
    
    results = retriever.fetch_video_from_db(results)

    end_time = time.perf_counter()
    q_time = end_time - start_time

    st.subheader("Results")
    st.write(f"Query time : {q_time:.4f} s")

    if len(results) == 0:
        st.write("No video match your query.")
    else:
        n_cols = 3 

        for i in range(0, len(results), n_cols):
            cols = st.columns(n_cols)
            for col, (id, dist, vid, metadata, label) in zip(cols, results[i:i+n_cols]):
                col.video(
                    vid,
                    caption=f"Similarity Score= {dist:.4f}",
                    width="stretch"
                )
                with col.expander("Metadata"):
                    st.write(
                        video_caption( *metadata)
                        )
                
