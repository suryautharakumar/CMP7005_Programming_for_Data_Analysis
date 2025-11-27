import streamlit as st
import math
import time

st.set_page_config(page_title="Multi App | Cardiff Metropolitan University", page_icon="https://www.cardiffmet.ac.uk/media/cardiff-met/site-assets/images/apple-touch-icon.png", layout="centered")

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
            background-image: url("https://img2.wallspic.com/crops/6/3/7/9/6/169736/169736-smartphone-drawing-android-painting-art-1500x3000.png");
            background-size: cover;
            background-repeat: no-repeat; 
            background-position: center; 
            color:#ffffff;
        }

        label[data-baseweb="radio"] span {
            color: #ffffff !important; 
        }
        input[type="radio"]:checked + label div:first-child {
            background-color: #ec64fc !important; /* Matches your label color */
            border-color: #ec64fc !important;
        }

    </style>
""", unsafe_allow_html=True)

#Sidebar
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="https://www.cardiffmet.ac.uk/media/cardiff-met/site-assets/images/apple-touch-icon.png" width="50" style="margin-right: 15px; border-radius:5rem;">
        <h1 style="font-size: 1.5em; margin: 0; color:#ffffff">Navigation Menu</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.write("Select a feature to explore:")
page = st.sidebar.radio(
    "",
    ("🏗️ Calculator", "⚖️ BMI Calculator", "📐 Area Calculator")
)
st.sidebar.markdown("---")
