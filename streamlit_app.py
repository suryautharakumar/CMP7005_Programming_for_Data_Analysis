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
    <div style="display: flex; align-items: center; margin-bottom:10px">
        <img src="https://www.cardiffmet.ac.uk/media/cardiff-met/site-assets/images/apple-touch-icon.png" width="50" style="margin-right: 15px; border-radius:50%;">
        <h1 style="font-size: 1.2em; margin: 0;">Navigation Bar</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.write("📈 Air Quality Data Analysis")
page = st.sidebar.radio(
    "Select a feature to explore:",
    ("⏳ Data Loading", "🧹 Data Pre processing", "Data Visualization")
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

    with st.spinner("⏳ Loading data..."):
        time.sleep(1.9)

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

if page == "🧹 Data Pre processing":

    st.header("🧹 Data Pre-processing")

    df_processed = df.copy()  # work on a fresh copy

    st.success("Data loaded for preprocessing!")

    # -----------------------------------------------------------
    # 1️⃣ HANDLE MISSING VALUES
    # -----------------------------------------------------------
    st.subheader("🚨 Handle Missing Values")

    handle_method = st.selectbox(
        "Choose a missing value handling method:",
        [
            "Do Nothing",
            "Drop rows with missing values",
            "Fill numeric (mean) & categorical (mode)",
            "Fill numeric (median)",
            "Fill ALL missing with custom value",
        ]
    )

    if handle_method == "Drop rows with missing values":
        df_processed = df_processed.dropna()
        st.info("✔ Rows with missing values dropped.")

    elif handle_method == "Fill numeric (mean) & categorical (mode)":
        num_cols = df_processed.select_dtypes(include=["float", "int"]).columns
        cat_cols = df_processed.select_dtypes(include=["object"]).columns

        df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].mean())
        df_processed[cat_cols] = df_processed[cat_cols].fillna(df_processed[cat_cols].mode().iloc[0])

        st.info("✔ Numeric filled with **mean**, categorical filled with **mode**.")

    elif handle_method == "Fill numeric (median)":
        num_cols = df_processed.select_dtypes(include=["float", "int"]).columns
        df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].median())
        st.info("✔ Numeric columns filled with **median**.")

    elif handle_method == "Fill ALL missing with custom value":
        custom_value = st.text_input("Enter a value to fill missing cells:")
        if custom_value != "":
            df_processed = df_processed.fillna(custom_value)
            st.info(f"✔ All missing cells filled with **{custom_value}**.")

    st.markdown("---")


    # -----------------------------------------------------------
    # 2️⃣ EXPLORE DATA DISTRIBUTIONS
    # -----------------------------------------------------------
    st.subheader("📊 Explore Data Distributions")

    numeric_cols = df_processed.select_dtypes(include=["float", "int"]).columns

    selected_dist_col = st.selectbox(
        "Choose a column to visualize distribution:",
        numeric_cols
    )

    dist_type = st.radio(
        "Select visualization type:",
        ["Histogram", "Boxplot"],
        horizontal=True
    )

    # Plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    if dist_type == "Histogram":
        ax.hist(df_processed[selected_dist_col].dropna(), bins=40)
        ax.set_title(f"Histogram - {selected_dist_col}")
    else:
        ax.boxplot(df_processed[selected_dist_col].dropna())
        ax.set_title(f"Boxplot - {selected_dist_col}")

    st.pyplot(fig)

    st.markdown("---")


    # -----------------------------------------------------------
    # 3️⃣ FEATURE ENGINEERING
    # -----------------------------------------------------------
    st.subheader("🛠 Feature Engineering")

    # Convert Date column
    df_processed["Date"] = pd.to_datetime(df_processed["Date"], errors="coerce")

    # Month column
    df_processed["Month"] = df_processed["Date"].dt.month

    # Season column
    def map_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df_processed["Season"] = df_processed["Month"].apply(map_season)

    st.success("✔ Month and Season columns added!")

    st.markdown("---")


    # -----------------------------------------------------------
    # 4️⃣ PREVIEW PROCESSED DATA (with spinner)
    # -----------------------------------------------------------
    st.subheader("👀 View Processed Data")

    prev_col1, prev_col2 = st.columns([2, 1])

    with prev_col1:
        rows_to_show = st.number_input(
            "Enter number of rows to preview:",
            min_value=1,
            max_value=len(df_processed),
            value=5,
            step=1
        )

    with prev_col2:
        st.write(f"📌 **Total Rows Available:** {len(df_processed)}")

    with st.spinner("⏳ Loading processed data..."):
        time.sleep(1.5)

    st.dataframe(df_processed.head(rows_to_show), use_container_width=True)

    st.markdown("---")

    st.success("🎉 Data Preprocessing Completed!")
