import os
import sys

import streamlit as st


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from Dashboard.model_comparison_page import render_model_comparison_page


st.set_page_config(page_title="Fraud Model Comparison Dashboard", layout="wide")
render_model_comparison_page(show_title=True)
