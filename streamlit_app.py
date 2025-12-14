import streamlit as st
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
            background-color: #ace177;
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
            background-color: #8ce3fd;
      
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
    ("⏳ Data Loading", "🧹 Data Pre processing", "📊 Data Visualization", "🧠 Data Prediction", "🧪 AQI Category Prediction")
)
st.sidebar.markdown("---")


#Data Loading Page

if page == "⏳ Data Loading":

    st.header("📄 Data Loading & Overview")

    #Load the dataset
    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/suryautharakumar/CMP7005_Programming_for_Data_Analysis/main/all_cities_combined.csv"
        df = pd.read_csv(url)
        return df

    df = load_data()

    #Saving dataset for global use
    st.session_state["df"] = df

    rows, cols = df.shape

    #Top Summary Metrics (Attractive Cards)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{rows:,}")
    col2.metric("Total Columns", cols)
    col3.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
    st.success("Dataset Loaded Successfully!")
    st.markdown("---")

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

        st.subheader("🏙 Total Missing Values per City")

        # Calculate missing values for each city
        city_missing_totals = df.groupby("City").apply(lambda x: x.isnull().sum().sum())
        city_missing_totals = city_missing_totals.reset_index()
        city_missing_totals.columns = ["City", "Total Missing Values"]

        # Color formatting
        city_total_style = city_missing_totals.style.background_gradient(cmap="Purples")
        st.dataframe(city_total_style, use_container_width=True)

        st.markdown("---")

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
            
# Data Pre Processing
if page == "🧹 Data Pre processing":

    st.header("🧹 Data Pre-processing")

    #check data loaded
    if "df" not in st.session_state:
        st.error("❌ Dataset not loaded. Go to '⏳ Data Loading' first.")
        st.stop()

    df = st.session_state["df"]

    if "df_processed" not in st.session_state:
        st.session_state["df_processed"] = df.copy()

    df_processed = st.session_state["df_processed"]

    #Message storage
    if "msg_missing" not in st.session_state:
        st.session_state["msg_missing"] = []

    if "msg_feature" not in st.session_state:
        st.session_state["msg_feature"] = []

    st.success("Data loaded for preprocessing!")

   #Handle missing values
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

    custom_value = None
    if handle_method == "Fill ALL missing with custom value":
        custom_value = st.text_input("Enter a value to fill missing cells:")

    if handle_method != "Do Nothing":
        if st.button("✔ Apply"):
            temp_df = df_processed.copy()

            if handle_method == "Drop rows with missing values":
                temp_df = temp_df.dropna()
                st.session_state["msg_missing"] = ["✔ Rows with missing values dropped."]

            elif handle_method == "Fill numeric (mean) & categorical (mode)":
                num_cols = temp_df.select_dtypes(include=["float", "int"]).columns
                cat_cols = temp_df.select_dtypes(include=["object"]).columns

                temp_df[num_cols] = temp_df[num_cols].fillna(temp_df[num_cols].mean())
                temp_df[cat_cols] = temp_df[cat_cols].fillna(temp_df[cat_cols].mode().iloc[0])

                st.session_state["msg_missing"] = ["✔ Numeric → mean | Categorical → mode."]

            elif handle_method == "Fill numeric (median)":
                num_cols = temp_df.select_dtypes(include=["float", "int"]).columns
                temp_df[num_cols] = temp_df[num_cols].fillna(temp_df[num_cols].median())
                st.session_state["msg_missing"] = ["✔ Numeric columns → median."]

            elif handle_method == "Fill ALL missing with custom value":
                if custom_value != "":
                    temp_df = temp_df.fillna(custom_value)
                    st.session_state["msg_missing"] = [
                        f"✔ Filled all missing cells with '{custom_value}'."
                    ]
                else:
                    st.warning("⚠ Please enter a custom value!")

            # Store updated DF
            st.session_state["df_processed"] = temp_df
            df_processed = temp_df

    for msg in st.session_state["msg_missing"]:
        st.success(msg)

    st.markdown("---")

    #EDA
    st.subheader("📊 Explore Data Distributions")

    numeric_cols = df_processed.select_dtypes(include=["float", "int"]).columns

    selected_dist_cols = st.multiselect(
        "Choose one or more columns:",
        numeric_cols
    )

    dist_type = st.radio(
        "Select visualization type:",
        ["Histogram", "Boxplot"],
        horizontal=True
    )

    if selected_dist_cols:
        for col in selected_dist_cols:
            st.write(f"### 📌 {col}")

            fig, ax = plt.subplots()

            if dist_type == "Histogram":
                ax.hist(df_processed[col].dropna(), bins=40)
                ax.set_title(f"Histogram - {col}")
            else:
                ax.boxplot(df_processed[col].dropna())
                ax.set_title(f"Boxplot - {col}")

            st.pyplot(fig)

    st.markdown("---")

    #feature Engineering
    st.subheader("🛠 Feature Engineering")
    st.write("Click to add **Month** & **Season** columns.")

    if st.button("⚙ Run Feature Engineering"):
        temp_df = df_processed.copy()

        temp_df["Date"] = pd.to_datetime(temp_df["Date"], errors="coerce")
        temp_df["Month"] = temp_df["Date"].dt.month

        def map_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Autumn"

        temp_df["Season"] = temp_df["Month"].apply(map_season)

        st.session_state["df_processed"] = temp_df
        df_processed = temp_df

        st.session_state["msg_feature"] = ["✔ Month & Season added successfully!"]

    #feature engineering message
    for msg in st.session_state["msg_feature"]:
        st.success(msg)

    st.markdown("---")

    #View Proced data
    st.subheader("👀 View Processed Data")

    if "show_preview" not in st.session_state:
        st.session_state["show_preview"] = False

    show_full = st.toggle("Show full dataset")

    rows_to_show = st.number_input(
        "Rows to preview:",
        min_value=1,
        max_value=len(df_processed),
        value=5,
        step=1
    )

    if st.button("👁 View Data"):
        st.session_state["show_preview"] = True

    if st.session_state["show_preview"]:
        with st.spinner("⏳ Loading processed data..."):
            time.sleep(1.8)

        if show_full:
            st.dataframe(df_processed, use_container_width=True)
        else:
            st.dataframe(df_processed.head(rows_to_show), use_container_width=True)

    st.markdown("---")
   
    #summary panel
    st.subheader("✨ Summary Panel")

    if st.button("📌 Show Summary"):
        st.write("### ❗ Missing Values")
        st.write(df_processed.isnull().sum())

        st.write("### 📈 Statistical Summary")
        summary_option = st.selectbox(
            "Select summary type:",
            ["Numeric Summary", "Categorical Summary"]
        )

        if summary_option == "Numeric Summary":
            st.write(df_processed.describe())
        else:
            st.write(df_processed.describe(include='object'))

    st.markdown("---")


if page == "📊 Data Visualization":

    st.header("📊 Data Visualization")

 
    if "df_processed" not in st.session_state:
        st.error("❌ No processed data found. Please complete preprocessing first. Missing values and feature engineering")
        st.stop()

    df = st.session_state["df_processed"]

   
    st.subheader("✨ Quick Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric Columns", len(df.select_dtypes(include=["float", "int"]).columns))
    col4.metric("Categorical Columns", len(df.select_dtypes(include=["object"]).columns))

    with st.expander("📋 Column Types"):
        st.write(df.dtypes)

    with st.expander("❗ Missing Values"):
        st.write(df.isnull().sum())

    st.markdown("---")

    st.subheader("🎨 Custom Visualization")

    chart_type = st.selectbox(
        "Select a chart type:",
        [
            "Line Chart",
            "Bar Chart",
            "Area Chart",
            "Scatter Plot",
            "Pie Chart",
            "Boxplot",
            "Correlation Heatmap",
        ]
    )

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    import matplotlib.pyplot as plt
    import numpy as np

    #line, area, bar
    if chart_type in ["Line Chart", "Area Chart", "Bar Chart"]:
        x_axis = st.selectbox("X-axis column:", df.columns)
        y_axis = st.multiselect("Y-axis column(s):", numeric_cols)

        if y_axis:
            if chart_type == "Line Chart":
                st.line_chart(df.set_index(x_axis)[y_axis])
            elif chart_type == "Area Chart":
                st.area_chart(df.set_index(x_axis)[y_axis])
            else:
                st.bar_chart(df.set_index(x_axis)[y_axis])

    #pie
    elif chart_type == "Pie Chart":
        pie_col = st.selectbox("Select categorical column:", categorical_cols)

        if pie_col:
            fig, ax = plt.subplots()
            counts = df[pie_col].value_counts()
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%")
            ax.set_title(f"Pie Chart of {pie_col}")
            st.pyplot(fig)

    #Scatter plot
    elif chart_type == "Scatter Plot":
        x = st.selectbox("X-axis (numeric):", numeric_cols)
        y = st.selectbox("Y-axis (numeric):", numeric_cols)
        color = st.selectbox("Group by (optional):", ["None"] + list(categorical_cols))

        fig, ax = plt.subplots()

        if color != "None":
            for v in df[color].unique():
                sub = df[df[color] == v]
                ax.scatter(sub[x], sub[y], label=v)
            ax.legend()
        else:
            ax.scatter(df[x], df[y])

        ax.set_title(f"{x} vs {y}")
        st.pyplot(fig)

    #box
    elif chart_type == "Boxplot":
        col = st.selectbox("Select numeric column:", numeric_cols)
        fig, ax = plt.subplots()
        ax.boxplot(df[col].dropna())
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

    #correlation
    elif chart_type == "Correlation Heatmap":
        corr = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr)
        fig.colorbar(cax)

        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45)
        ax.set_yticklabels(numeric_cols)

        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

    st.markdown("---")

    #Adv Vis
    st.subheader("🚀 Advanced Visualizations")

    adv_chart = st.selectbox(
        "Choose advanced option:",
        [
            "Scatter Matrix",
            "Grouped Aggregation",
            "Outlier Detection",
        ]
    )

    #scatter
    if adv_chart == "Scatter Matrix":
        from pandas.plotting import scatter_matrix

        selected_cols = st.multiselect("Select up to 5 numeric columns:", numeric_cols)

        if len(selected_cols) > 0:
            fig = scatter_matrix(df[selected_cols], figsize=(10, 8))
            st.pyplot(plt.gcf())

    #Grouped Agg
    elif adv_chart == "Grouped Aggregation":
        group_col = st.selectbox("Group by (categorical):", categorical_cols)
        agg_col = st.selectbox("Aggregate column (numeric):", numeric_cols)
        func = st.selectbox("Aggregation function:", ["mean", "sum", "count"])

        result = df.groupby(group_col)[agg_col].agg(func)
        st.bar_chart(result)


    #outlier
    elif adv_chart == "Outlier Detection":
        metric_col = st.selectbox("Select numeric column:", numeric_cols)

        Q1 = df[metric_col].quantile(0.25)
        Q3 = df[metric_col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[metric_col] < lower) | (df[metric_col] > upper)]

        st.warning(f"Found {len(outliers)} outliers in **{metric_col}**")

        fig, ax = plt.subplots()
        ax.boxplot(df[metric_col].dropna())
        ax.set_title(f"Outlier Detection for {metric_col}")
        st.pyplot(fig)

        with st.expander("Show Outlier Rows"):
            st.write(outliers)


if page == "🧠 Data Prediction":

    st.header("🧠 Modelling & Prediction")

   
    if "df_processed" not in st.session_state:
        st.error("❌ Please complete Data Pre-processing first. Missing values and feature engineering")
        st.stop()

    df = st.session_state["df_processed"].copy()

    st.success("Processed dataset loaded successfully!")

    st.subheader("🎯 Feature & Target Selection")

    target_col = "AQI"

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()

    if target_col not in numeric_cols:
        st.error("❌ AQI column missing or not numeric.")
        st.stop()

    feature_cols = [col for col in numeric_cols if col != target_col]

    selected_features = st.multiselect(
        "Select input features:",
        feature_cols,
        default=feature_cols[:6]
    )

    if len(selected_features) == 0:
        st.warning("⚠ Please select at least one feature.")
        st.stop()

    X = df[selected_features]
    y = df[target_col]

#test train
    st.subheader("✂ Train-Test Split")

    test_size = st.slider("Test size (%)", 10, 40, 20) / 100

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    st.info(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")

#comparison
    st.markdown("---")
    st.subheader("📊 Model Comparison")

    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import numpy as np
    import pandas as pd

    if "model_comparison" not in st.session_state:
        st.session_state["model_comparison"] = None

    if st.button("📊 Compare Models"):

        with st.spinner("Training models and evaluating..."):

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
                "Random Forest": RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )
            }

            results = []

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                results.append({
                    "Model": name,
                    "R² Score": round(r2_score(y_test, preds), 3),
                    "MAE": round(mean_absolute_error(y_test, preds), 2),
                    "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2)
                })

            results_df = pd.DataFrame(results).sort_values(
                "R² Score", ascending=False
            )

            st.session_state["model_comparison"] = results_df

    #display
    if st.session_state["model_comparison"] is not None:

        st.subheader("📋 Model Performance Comparison")

        styled_df = (
            st.session_state["model_comparison"]
            .style
            .highlight_max(subset=["R² Score"], color="#90EE90")
            .highlight_min(subset=["MAE", "RMSE"], color="#90EE90")
        )

        st.dataframe(styled_df, use_container_width=True)

        best_model_name = st.session_state["model_comparison"].iloc[0]["Model"]
        best_r2 = st.session_state["model_comparison"].iloc[0]["R² Score"]

        st.success(f"🏆 Best Model: **{best_model_name}** (R² = {best_r2})")

    #Train best model
    st.markdown("---")
    st.subheader("🔮 AQI Prediction")

    model_map = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    chosen_model_name = st.selectbox(
        "Select model for prediction:",
        list(model_map.keys())
    )

    model = model_map[chosen_model_name]
    model.fit(X_train, y_train)

    st.markdown("### 🧪 Enter feature values")

    user_input = {}
    for col in selected_features:
        user_input[col] = st.number_input(
            f"{col}",
            value=float(X[col].mean())
        )

    if st.button("📈 Predict AQI"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]

        st.success(f"🌍 **Predicted AQI:** {round(prediction, 2)}")

    st.markdown("---")
    st.info("✅ Modelling & Prediction completed successfully.")


if page == "🧪 AQI Category Prediction":

    st.header("🧪 AQI Category Prediction (Classification)")

    # -----------------------------------------------------------
    # CHECK DATA
    # -----------------------------------------------------------
    if "df_processed" not in st.session_state:
        st.error("❌ Please complete Data Pre-processing first.")
        st.stop()

    df = st.session_state["df_processed"].copy()

    if "AQI_Bucket" not in df.columns:
        st.error("❌ AQI_Bucket column not found.")
        st.stop()

    st.success("Processed dataset loaded successfully!")

    # -----------------------------------------------------------
    # DATA PREPARATION
    # -----------------------------------------------------------
    st.subheader("📂 Data Preparation")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import numpy as np

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "AQI"]

    target = "AQI_Bucket"

    X = df[numeric_cols].fillna(df[numeric_cols].median())
    y = df[target]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    st.info(f"Features used: {len(numeric_cols)} | Classes: {len(label_encoder.classes_)}")

    # -----------------------------------------------------------
    # TRAIN-TEST SPLIT
    # -----------------------------------------------------------
    test_size = st.slider("Test size (%)", 10, 40, 20) / 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )

    # -----------------------------------------------------------
    # MODEL COMPARISON
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Classification Model Comparison")

    if "clf_results" not in st.session_state:
        st.session_state["clf_results"] = None

    if st.button("📊 Train & Compare Models"):

        with st.spinner("Training classification models..."):

            classifiers = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(max_depth=6),
                "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42)
            }

            results = []

            for name, clf in classifiers.items():
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)

                acc = accuracy_score(y_test, preds)

                results.append({
                    "Model": name,
                    "Accuracy": round(acc, 3)
                })

            results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
            st.session_state["clf_results"] = results_df

    # -----------------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------------
    if st.session_state["clf_results"] is not None:

        st.subheader("📋 Model Accuracy Comparison")

        styled_df = (
            st.session_state["clf_results"]
            .style
            .highlight_max(subset=["Accuracy"], color="#90EE90")
        )

        st.dataframe(styled_df, use_container_width=True)

        best_model_name = st.session_state["clf_results"].iloc[0]["Model"]
        st.success(f"🏆 Best Model: **{best_model_name}**")

    # -----------------------------------------------------------
    # DETAILED CLASSIFICATION REPORT
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("📑 Classification Report")

    model_choice = st.selectbox(
        "Select model for detailed evaluation:",
        ["Logistic Regression", "Decision Tree", "Random Forest"]
    )

    model_map = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42)
    }

    clf = model_map[model_choice]
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    report = classification_report(
        y_test,
        preds,
        target_names=label_encoder.classes_,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    # -----------------------------------------------------------
    # SINGLE AQI CATEGORY PREDICTION
    # -----------------------------------------------------------
    st.markdown("---")
    st.subheader("🔮 Predict AQI Category")

    st.markdown("### 🧪 Enter pollutant values")

    user_input = {}
    for col in numeric_cols:
        user_input[col] = st.number_input(
            f"{col}",
            value=float(X[col].mean())
        )

    if st.button("🧠 Predict AQI Category"):
        input_df = pd.DataFrame([user_input])
        pred_class = clf.predict(input_df)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]

        st.success(f"🌍 **Predicted AQI Category:** {pred_label}")

    st.markdown("---")
    st.info("✅ AQI Category Classification completed successfully.")
