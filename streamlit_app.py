import streamlit as st
import math
import time
import pandas as pd
import io


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

    st.header("📄 Data Loading & Overview")

    # Load the dataset
    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/suryautharakumar/CMP7005_Programming_for_Data_Analysis/main/all_cities_combined.csv"
        df = pd.read_csv(url)
        return df

    df = load_data()
    rows, cols = df.shape

    # Top Summary Metrics (Attractive Cards)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{rows:,}")
    col2.metric("Total Columns", cols)
    col3.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
    st.success("Dataset Loaded Successfully!")
    st.markdown("---")

    # ---------------- USER ROW PREVIEW REQUEST ----------------
    st.subheader("🔍 Preview Dataset")

    preview_col1, preview_col2 = st.columns([2, 1])

    with preview_col1:
        num_rows = st.number_input(
            "Enter number of rows to preview:",
            min_value=1,
            max_value=rows,
            value=5,
            step=1,
        )

    with preview_col2:
        st.write(f"📌 **Total Rows Available:** {rows}")

    st.dataframe(df.head(num_rows), use_container_width=True)

    st.markdown("---")

    # ----------------- EXPANDERS -------------------

    # Column Data Types
    with st.expander("🧬 Column Data Types"):
        st.dataframe(df.dtypes.to_frame("Data Type"))

    # Missing Values
    with st.expander("⚠ Missing Values Summary"):
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        missing_df["Missing %"] = round((missing_df["Missing Values"] / len(df)) * 100, 2)
        
        # Color formatting for readability
        missing_df = missing_df.style.background_gradient(cmap="Oranges")
        st.dataframe(missing_df, use_container_width=True)

    # Dataset Information (df.info)
    with st.expander("ℹ Dataset Information (df.info)"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)

    # Basic Statistics
    with st.expander("📊 Statistical Summary"):
        st.dataframe(df.describe(include="all"), use_container_width=True)

    # Unique Categoricals
    with st.expander("🔠 Unique Values (Categorical Columns)"):
        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            st.write(f"### {col}")
            st.write(f"🟣 Unique Values: **{df[col].nunique()}**")
            st.write(df[col].unique())
            st.markdown("---")
