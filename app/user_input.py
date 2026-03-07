import streamlit as st

def get_user_details():
    st.subheader("User Information")

    name = st.text_input("Enter your Name")
    age = st.number_input("Enter Age", min_value=1, max_value=120)
    gender = st.selectbox("Select Gender", ["Male", "Female", "Other"])



    return name, age, gender