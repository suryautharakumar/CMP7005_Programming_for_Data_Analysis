# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# ----------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------
st.set_page_config(page_title="Air Quality EDA Dashboard", layout="wide")

st.title("🌍 Air Quality EDA Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("raw.githubusercontent.com/suryautharakumar/CMP7005_Programming_for_Data_Analysis/refs/heads/main/all_cities_combined.csv", parse_dates=["Date"])
    return df

df = load_data()

# ----------------------------------------------------------------------
# CITY COORDINATES
# ----------------------------------------------------------------------
city_coords = pd.DataFrame({
    'City': [
        'Kochi','Bhopal','Lucknow','Kolkata','Guwahati','Amaravati','Aizawl',
        'Thiruvananthapuram','Jaipur','Ernakulam','Bengaluru','Mumbai','Hyderabad',
        'Patna','Talcher','Shillong','Coimbatore','Visakhapatnam','Chennai',
        'Gurugram','Brajrajnagar','Amritsar','Jorapokhar','Delhi','Chandigarh','Ahmedabad'
    ],
    'Latitude': [
        9.9312,23.2599,26.8467,22.5726,26.1445,16.5062,23.7271,
        8.5241,26.9124,9.9816,12.9716,19.0760,17.3850,
        25.5941,20.9485,25.5788,11.0168,17.6868,13.0827,
        28.4595,21.8100,31.6340,23.7975,28.7041,30.7333,23.0225],
    'Longitude': [
        76.2673,77.4126,80.9462,88.3639,91.7362,80.6480,92.7176,
        76.9366,75.7873,76.2999,77.5946,72.8777,78.4867,
        85.1376,85.1950,91.8933,76.9558,83.2185,80.2707,
        77.0266,83.1666,74.8723,86.1955,77.1025,76.7794,72.5714]
})

df = df.merge(city_coords, on="City", how="left")

# ----------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------
def aqi_color(aqi):
    if aqi <= 50: return "green"
    elif aqi <= 100: return "lightblue"
    elif aqi <= 150: return "orange"
    elif aqi <= 200: return "red"
    elif aqi <= 300: return "purple"
    else: return "black"

def aqi_radius(aqi):
    return min(25, max(4, np.log1p(aqi)))

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
st.sidebar.header("Dashboard Controls")

selected_cities = st.sidebar.multiselect(
    "Select Cities",
    options=df["City"].unique(),
    default=df["City"].unique()
)

df_filtered = df[df["City"].isin(selected_cities)]

# ----------------------------------------------------------------------
# (2) FOLIUM MAP
# ----------------------------------------------------------------------
st.subheader("🗺️ Air Quality Map")

def create_map(df_city_summary):
    center_lat = df_city_summary["Latitude"].mean()
    center_lon = df_city_summary["Longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='CartoDB positron')
    cluster = MarkerCluster().add_to(m)

    for _, row in df_city_summary.iterrows():
        popup_html = f"""
        <b>{row['City']}</b><br>
        AQI: {row['AQI']:.1f}<br>
        PM2.5: {row['PM2.5']:.1f}<br>
        PM10: {row['PM10']:.1f}<br>
        """

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=aqi_radius(row["AQI"]),
            color=aqi_color(row["AQI"]),
            fill=True,
            fill_opacity=0.6,
            popup=popup_html,
            tooltip=row["City"]
        ).add_to(cluster)

    HeatMap(df_city_summary[["Latitude","Longitude","AQI"]].values.tolist(), radius=25).add_to(m)
    return m

city_summary = df_filtered.groupby("City").agg({
    "Latitude":"first", "Longitude":"first",
    "AQI":"mean", "PM2.5":"mean", "PM10":"mean"
}).reset_index()

map_obj = create_map(city_summary)
st_folium(map_obj, width=1100, height=550)

# ----------------------------------------------------------------------
# (4) RADAR CHART
# ----------------------------------------------------------------------
st.subheader("📊 City Pollution Radar Chart")

pollutants = ['PM2.5','PM10','NO2','O3','SO2','CO']

def radar_chart(df, cities):
    fig = go.Figure()
    for city in cities:
        city_vals = df[df["City"] == city][pollutants].mean().values
        fig.add_trace(go.Scatterpolar(
            r=list(city_vals) + [city_vals[0]],
            theta=pollutants + [pollutants[0]],
            fill='toself',
            name=city
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        legend=dict(orientation="h"),
        height=600
    )
    return fig

st.plotly_chart(radar_chart(df_filtered, selected_cities), use_container_width=True)

# ----------------------------------------------------------------------
# (5) AQI TIME SERIES DASHBOARD
# ----------------------------------------------------------------------
st.subheader("📈 AQI Time Series")

resample_option = st.selectbox("Resample frequency", ["Daily", "Monthly"])
rolling = st.slider("Rolling average (days)", 1, 30, 10)

resample_map = {"Daily":"D", "Monthly":"M"}
rule = resample_map[resample_option]

fig_ts = go.Figure()
for city in selected_cities:
    s = df[df["City"]==city].set_index("Date")["AQI"].resample(rule).mean()
    fig_ts.add_trace(go.Scatter(x=s.index, y=s, name=f"{city} mean"))
    fig_ts.add_trace(go.Scatter(x=s.index, y=s.rolling(rolling).mean(), name=f"{city} {rolling}-day roll"))

fig_ts.update_layout(height=500, hovermode="x unified")
st.plotly_chart(fig_ts, use_container_width=True)

# ----------------------------------------------------------------------
# (7) OUTLIER ANALYSIS
# ----------------------------------------------------------------------
st.subheader("🚨 Outlier Detection")

method = st.radio("Select method", ["Z-score", "IQR"])
pollutant_col = st.selectbox("Select pollutant", pollutants)

if method == "Z-score":
    z = np.abs(stats.zscore(df[pollutant_col].fillna(df[pollutant_col].median())))
    outliers = df[z > 3]
else:
    Q1 = df[pollutant_col].quantile(0.25)
    Q3 = df[pollutant_col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[pollutant_col] < Q1 - 1.5*IQR) | (df[pollutant_col] > Q3 + 1.5*IQR)]

st.write(f"Found **{len(outliers)}** outliers.")

st.dataframe(outliers.head())

# Download button
outliers_csv = outliers.to_csv(index=False).encode()
st.download_button("Download outliers as CSV", outliers_csv, "outliers.csv")

# Boxplot with outliers
fig_box = px.box(df, y=pollutant_col, points="outliers", height=400)
st.plotly_chart(fig_box, use_container_width=True)

# ----------------------------------------------------------------------
# END OF APP
# ----------------------------------------------------------------------


