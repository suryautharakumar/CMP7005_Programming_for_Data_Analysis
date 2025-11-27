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
            background-image: url("https://raw.githubusercontent.com/suryautharakumar/CMP7005_Programming_for_Data_Analysis/refs/heads/main/dl.beatsnoop.com-3000-xPqy47jcDi.jpg");
            background-size: cover;
            background-repeat: no-repeat; 
            background-position: center;
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
    ("⏳ Data Loading", "Data Pre processing", "Data Visualization")
)
st.sidebar.markdown("---")


#Data Loading Page

if page == "⏳ Data Loading":

    import streamlit as st
    import pandas as pd
    import io

    st.header("📄 Data Loading")

    # Load the dataset
    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/suryautharakumar/CMP7005_Programming_for_Data_Analysis/main/all_cities_combined.csv"
        df = pd.read_csv(url)
        return df

    df = load_data()

    st.success("Dataset Loaded Successfully!")

    # ----- Interactive Components -----

    # 1) Preview dataset
    with st.expander("🔍 Preview Dataset"):
        st.write(df.head(10))

    # 2) Dataset Shape
    with st.expander("📏 Dataset Shape"):
        rows, cols = df.shape
        st.write(f"**Rows:** {rows}")
        st.write(f"**Columns:** {cols}")

    # 3) Column Data Types
    with st.expander("🧬 Column Data Types"):
        st.write(df.dtypes)

    # 4) Missing Values
    with st.expander("⚠ Missing Values Summary"):
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        missing_df["Missing %"] = round((missing_df["Missing Values"] / len(df)) * 100, 2)
        st.dataframe(missing_df)

    # 5) Info-like output
    with st.expander("ℹ Dataset Information"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    # 6) Basic Statistics
    with st.expander("📊 Basic Statistical Summary"):
        st.write(df.describe(include="all"))
        
    # 7) Unique values per categorical column
    with st.expander("🔠 Unique Values in Categorical Columns"):
        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            st.write(f"**{col}**: {df[col].nunique()} unique values")
            st.write(df[col].unique())
