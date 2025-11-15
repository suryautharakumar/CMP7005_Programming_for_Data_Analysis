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
            background-image: url("https://img.freepik.com/free-vector/colorful-gradient-background-modern-design_361591-4583.jpg?semt=ais_incoming&w=740&q=80");
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
        <img src="https://www.cardiffmet.ac.uk/media/cardiff-met/site-assets/images/apple-touch-icon.png" width="50" style="margin-right: 15px;">
        <h1 style="font-size: 1.5em; margin: 0;">Navigation Menu</h1>
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

#Calculator Page
if page == "🏗️ Calculator":
    st.title("🧮 Smart Calculator")
    st.write("Perform basic arithmetic and interest calculations instantly.")

    st.markdown("### 🔢 Enter Your Values")

    with st.container():
        col1, col2 = st.columns(2)
        num1 = col1.number_input("Enter the first number:", value=None)
        num2 = col2.number_input("Enter the second number:", value=None)

    operation = st.selectbox(
        "Select an operation:",
        (
            "Addition (+)",
            "Subtraction (-)",
            "Multiplication (×)",
            "Division (÷)",
            "Simple Interest",
            "Compound Interest",
            "Square",
            "Square Root"
        )
    )

    if st.button("✨ Calculate"):
        with st.spinner("Calculating..."):
            time.sleep(1.8)
        if operation == "Addition (+)":
            result = num1 + num2
            st.success(f"✅ Result: **{result}**")

        elif operation == "Subtraction (-)":
            result = num1 - num2
            st.success(f"✅ Result: **{result}**")

        elif operation == "Multiplication (×)":
            result = num1 * num2
            st.success(f"✅ Result: **{result}**")

        elif operation == "Division (÷)":
            if num2 == 0:
                st.error("❌ Cannot divide by zero!")
            else:
                result = num1 / num2
                st.success(f"✅ Result: **{result}**")

        elif operation == "Simple Interest":
            principal = num1
            rate = num2
            time_years = st.number_input("Enter time (in years):", value=1)
            si = (principal * rate * time_years) / 100
            st.info(f"💰 Simple Interest = **{si:.2f}**")

        elif operation == "Compound Interest":
            principal = num1
            rate = num2
            time_years = st.number_input("Enter time (in years):", value=1)
            ci = principal * ((1 + rate / 100) ** time_years) - principal
            st.info(f"💰 Compound Interest = **{ci:.2f}**")

        elif operation == "Square":
            result = num1 ** 2
            st.success(f"🟦 Square of {num1} = **{result}**")

        elif operation == "Square Root":
            result = math.sqrt(num1)
            st.success(f"🌀 Square Root of {num1} = **{result:.2f}**")

#BMI Calculator Page
elif page == "⚖️ BMI Calculator":
    st.title("⚖️ BMI Calculator")

    st.write("Use this calculator to find your **Body Mass Index (BMI)** and get a quick health assessment.")

    st.markdown("### 📏 Enter Your Details")

    height_cm = st.number_input("Enter your height (in centimeters):", value=None, min_value=0.0)
    weight = st.number_input("Enter your weight (in kilograms):", value=None, min_value=0.0)

    if st.button("💡 Calculate BMI"):
        if height_cm <= 0:
            st.error("⚠️ Height must be greater than 0.")
        else:
            with st.spinner("calculating BMI..."):
                time.sleep(1.8)
            height_m = height_cm / 100
            bmi = weight / (height_m ** 2)
            st.write(f"### Your BMI is: **{bmi:.2f}**")

            if bmi < 18.5:
                st.warning("🟡 You are **Underweight** — eat healthy and gain some mass!")
            elif 18.5 <= bmi < 24.9:
                st.success("🟢 You have a **Normal Weight** — great job maintaining balance!")
            elif 25 <= bmi < 29.9:
                st.info("🔵 You are **Overweight** — consider a bit more physical activity.")
            else:
                st.error("🔴 You are **Obese** — focus on diet and exercise.")

#Area Calculator Page
elif page == "📐 Area Calculator":
    st.title("📐 Area Calculator")
    st.write("Compute the area of basic geometric shapes with ease.")

    st.markdown("### 🧩 Select a Shape")
    shape = st.selectbox("Choose shape:", ("Circle", "Rectangle", "Triangle"))

    if shape == "Circle":
        radius = st.number_input("Enter the radius:", value=None, min_value=0.0)
        if st.button("Calculate Circle Area"):
            with st.spinner("Calculating..."):
                time.sleep(1.8)
            area = math.pi * (radius ** 2)
            st.success(f"🟣 Area of Circle = **{area:.2f}** sq. units")

    elif shape == "Rectangle":
        length = st.number_input("Enter the length:", value=None, min_value=0.0)
        width = st.number_input("Enter the width:", value=None, min_value=0.0)
        if st.button("Calculate Rectangle Area"):
            with st.spinner("Calculating..."):
                time.sleep(1.8)
            area = length * width
            st.success(f"🟩 Area of Rectangle = **{area:.2f}** sq. units")

    elif shape == "Triangle":
        base = st.number_input("Enter the base:", value=None, min_value=0.0)
        height = st.number_input("Enter the height:", value=None, min_value=0.0)
        if st.button("Calculate Triangle Area"):
            with st.spinner("Calculating..."):
                time.sleep(1.8)
            area = 0.5 * base * height
            st.success(f"🔺 Area of Triangle = **{area:.2f}** sq. units")

#Footer
st.markdown("---")
