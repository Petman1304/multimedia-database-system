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



