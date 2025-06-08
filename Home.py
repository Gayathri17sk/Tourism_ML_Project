import streamlit as st
import base64

def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


set_background(r"C:\Users\Dell\Downloads\Download premium vector of Hand drawn travel element background vector set by marinemynt about background, cartoon, paper, compass, and cute 936632.jpg")

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.title("🧳 Tourism Experience Analytics")
st.write("Predict Smarter, Explore Better. Your AI-powered travel planner.")


st.write("")   #enter option
st.write("")
st.write("")

st.subheader("About us🗺️")

st.markdown("""
##### **Tourism Experience Analytics** is a smart travel application designed to predict **ratings**, classify **visit modes**, and recommend **tourist attractions** using user data and machine learning.

This project involves:
- 🧹 **Data Cleaning**: Removing missing values, fixing formats, and resolving inconsistencies.
- 📊 **EDA**: Understanding trends in visit types, ratings, and destination popularity.
- 🧠 **Visit Mode Classification**: Predicting whether a user travels for **Family**, **Business**, or other purposes.
- ⭐ **Rating Prediction**: Estimating how users rate attractions based on demographics and past visits.
- 🎯 **Attraction Recommendation**:
  - 📌 **Content-Based**: Based on the user’s travel history.
  - 👥 **Collaborative Filtering**: Based on similar users’ preferences.

##### **Built with 🧳 Streamlit | 🤖 Machine Learning | 🌍 Clean Tourism Data**
""")

