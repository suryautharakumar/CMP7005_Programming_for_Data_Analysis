import streamlit as st
import math
import time

st.set_page_config(page_title="All Cities Air Quality Data Analysis | Cardiff Metropolitan University", page_icon="https://www.cardiffmet.ac.uk/media/cardiff-met/site-assets/images/apple-touch-icon.png", layout="centered")

#header css styling
st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
        }
        h1, h2, h3 {
            color: #b2e5f8;
        }
        .stButton>button {
            background-color: #eabbf1;
            color: #000000;
            border-radius: 10px;
            height: 2.8em;
            width: 100%;
            font-size: 1em;
            font-weight: bold;
            transition: 0.3s;
            transform: scale(1);
        }
        .stButton>button:hover {
            background-color: #b2e5f8;
      
            transform: scale(0.9);
        }
        .stSelectbox label, .stNumberInput label {
            font-weight: 800;
            font-weight: bold;
            color: #ec64fc;
        }
        [data-testid="stSidebar"] {
            background-image: url("https://media.istockphoto.com/id/481585470/vector/summer-vacation.jpg?s=2048x2048&w=is&k=20&c=O-Nz3f69lsFWfqSaBeuQA6q40qD0oEUwrSVOiaEcs1o=");
            background-size: cover;
            background-repeat: no-repeat; 
            background-position: lift;
        }
    </style>
""", unsafe_allow_html=True)

#Sidebar
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="https://www.cardiffmet.ac.uk/media/cardiff-met/site-assets/images/apple-touch-icon.png" width="50" style="margin-right: 15px; border-radius:50%;">
        <h1 style="font-size: 1.5em; margin: 0;">Navigation</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.write("Select a feature to explore:")
page = st.sidebar.radio(
    "",
    ("Data Loading", "Data Pre processing", "Data Visualization")
)
st.sidebar.markdown("---")
