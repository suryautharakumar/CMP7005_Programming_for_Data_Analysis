import streamlit as st
import math
import time
import pandas as pd
import io

# 79d125 - green
# 06d4fc - blue

st.set_page_config(page_title="All Cities Air Quality Data Analysis | Cardiff Metropolitan University", page_icon="https://www.cardiffmet.ac.uk/media/cardiff-met/site-assets/images/apple-touch-icon.png", layout="centered")

#header css styling
st.markdown("""
    <style>
        .main {
            background-color: #f8fafc;
        }
        h1, h2, h3 {
            color: #79d125;
        }
        .stButton>button {
            background-color: #06d4fc;
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
            background-color: #79d125;
      
            transform: scale(0.9);
        }
        .stSelectbox label, .stNumberInput label {
            font-weight: 900;
            font-weight: bold;
            color: #79d125;
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
    <div style="display: flex; align-items: center;">
        <img src="https://www.cardiffmet.ac.uk/media/cardiff-met/site-assets/images/apple-touch-icon.png" width="50" style="margin-right: 15px; border-radius:50%;">
        <h1 style="font-size: 1.5em; margin: 0;">Data Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.write("Select a feature to explore:")
page = st.sidebar.radio(
    "Select a feature to explore:",
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
        st.subheader("📌 Overall Missing Values (Entire Dataset)")
        
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        missing_df["Missing %"] = round((missing_df["Missing Values"] / len(df)) * 100, 2)

        overall_style = missing_df.style.background_gradient(cmap="Oranges")
        st.dataframe(overall_style, use_container_width=True)

        st.markdown("---")

        # ---------------- TOTAL MISSING VALUES PER CITY ----------------
        st.subheader("🏙 Total Missing Values per City")

        # Calculate missing values for each city
        city_missing_totals = df.groupby("City").apply(lambda x: x.isnull().sum().sum())
        city_missing_totals = city_missing_totals.reset_index()
        city_missing_totals.columns = ["City", "Total Missing Values"]

        # Color formatting
        city_total_style = city_missing_totals.style.background_gradient(cmap="Purples")
        st.dataframe(city_total_style, use_container_width=True)

        st.markdown("---")

        # ---------------- Missing Values per City (Detailed) ----------------
        st.subheader("🧩 Missing Values by Column for Each City")

        cities = df["City"].unique()
        selected_city = st.selectbox("Select a City to inspect missing values:", cities)

        city_df = df[df["City"] == selected_city]

        city_missing = city_df.isnull().sum().reset_index()
        city_missing.columns = ["Column", "Missing Values"]
        city_missing["Missing %"] = round((city_missing["Missing Values"] / len(city_df)) * 100, 2)

        city_style = city_missing.style.background_gradient(cmap="Blues")
        st.dataframe(city_style, use_container_width=True)

        st.info(f"📌 Total rows for **{selected_city}**: {len(city_df)}")



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
