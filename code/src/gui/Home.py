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

st.write(
    """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. In non ligula ut felis cursus congue. Proin in felis sit amet dui sagittis feugiat. Curabitur vel nunc cursus, lobortis orci vitae, mollis odio. Mauris pretium dictum bibendum. Quisque sollicitudin diam dui, eu tincidunt lacus blandit vehicula. Fusce et urna a neque condimentum vehicula sit amet a dui. Proin egestas orci at leo posuere rutrum. Phasellus velit urna, efficitur consectetur dolor a, maximus ultrices elit. Aenean imperdiet non magna ac tempor. Integer mattis arcu nec aliquam interdum.

Proin ac elit quis est placerat tincidunt. Duis sit amet cursus tortor, et ultrices massa. Mauris tincidunt quam ut felis sollicitudin consequat. Duis mattis libero magna, a consectetur eros rutrum ut. Donec ut erat feugiat enim efficitur euismod rhoncus tincidunt ligula. Aliquam tempor pulvinar commodo. Etiam imperdiet augue sapien, nec posuere metus tincidunt eu. Praesent quis risus diam. Integer semper nibh quis dolor fermentum pretium. Ut porttitor ligula nibh, eget feugiat felis scelerisque sed. Cras euismod vulputate tristique. Curabitur vel lorem interdum, porta dui nec, suscipit velit. Pellentesque non metus semper, sagittis dui non, sollicitudin arcu.

Donec finibus sem auctor odio hendrerit venenatis. Proin et luctus felis. Quisque blandit leo ac tellus condimentum, nec viverra nulla semper. Duis euismod ex non tempus convallis. Mauris eu nunc ultrices, pellentesque felis sed, faucibus sapien. Etiam dapibus consequat nisi, nec tempus turpis. Duis auctor dignissim lacus, eu tempus ipsum luctus vitae. Vestibulum eget elit mattis, sodales velit vitae, tristique ante. Duis non elementum mauris.

Curabitur quis metus tristique, vehicula nunc id, aliquet magna. Integer blandit, lorem a mollis blandit, elit arcu viverra risus, id pharetra purus nulla luctus est. Nullam efficitur eros id nibh posuere, vel laoreet ligula cursus. Vivamus faucibus odio quis sagittis aliquam. Sed venenatis blandit ex, eget accumsan leo faucibus a. Sed id mauris porttitor, imperdiet lectus fringilla, suscipit tortor. Morbi sit amet nisi diam. Cras maximus aliquam justo quis maximus. Morbi luctus varius neque non elementum. Aenean ac quam vitae nibh laoreet sagittis. Integer efficitur ullamcorper nisl eu commodo.

Morbi finibus condimentum neque, ac elementum libero molestie et. Duis nec maximus justo. Cras a nunc finibus, aliquam erat quis, porta ex. Quisque et porta ipsum, sit amet porta ex. Cras tincidunt, lacus eu tristique tristique, nunc enim ultrices dolor, sed congue ante est nec eros. Etiam efficitur lectus eu odio rhoncus sollicitudin. Duis viverra orci efficitur nunc iaculis interdum.
    """
    )



