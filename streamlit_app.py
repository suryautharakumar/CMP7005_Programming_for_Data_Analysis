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

    st.title("📥 Data Loading")

    st.subheader("Load Air Quality Dataset")
    load_btn = st.button("Load Dataset")

    if load_btn:
        with st.spinner("Loading data..."):
            df = pd.read_csv(
                "https://raw.githubusercontent.com/suryautharakumar/CMP7005_Programming_for_Data_Analysis/refs/heads/main/all_cities_combined.csv",
                parse_dates=["Date"]
            )
            st.success("Data Loaded Successfully!")
            st.write("### 🔍 Dataset Preview")
            st.dataframe(df.head())

            # BASIC INFO SECTION
            st.markdown("---")
            st.header("📊 Basic Information")

            # Shape
            with st.expander("📏 Dataset Shape"):
                st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")

            # Column names
            with st.expander("📋 Column Names"):
                st.write(df.columns.tolist())

            # Missing values
            with st.expander("❗ Missing Values"):
                missing = df.isnull().sum()
                st.write(missing)

                # Plot missing values
                fig, ax = plt.subplots(figsize=(10, 5))
                missing.plot(kind='bar', ax=ax)
                ax.set_title("Missing Values per Column")
                ax.set_ylabel("Count")
                st.pyplot(fig)

            # Info (convert df.info() to string)
            with st.expander("ℹ️ Dataset Info"):
                buffer = []
                df.info(buf=buffer)
                info_str = "\n".join(buffer)
                st.text(info_str)

            # Describe
            with st.expander("📈 Statistical Summary"):
                st.write(df.describe())

            # Extra: unique cities
            with st.expander("🏙️ Cities Included"):
                st.write(df["City"].unique())
