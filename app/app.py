import sys
import os
import streamlit as st
import base64
import pandas as pd
import plotly.express as px

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from user_input import get_user_details
from upload import upload_images
from predict import predict_parkinson

st.set_page_config(page_title="Parkinson Detection System", layout="wide")

# ---------------- BACKGROUND FUNCTION ----------------
def set_bg():
    bg_path = os.path.join(os.path.dirname(__file__), "images", "bg.jpg")

    if not os.path.exists(bg_path):
        st.warning("Background image not found.")
        return

    with open(bg_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
        background-image: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
        url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        }}

        h1,h2,h3,p {{
        color:white;
        }}

        .glass-container {{
        background: rgba(255,255,255,0.08);
        padding:50px;
        border-radius:20px;
        backdrop-filter: blur(12px);
        text-align:center;
        margin-top:100px;
        }}

        .glass-container h1 {{
            font-size:55px;
            font-weight:bold;
        }}

        .glass-container p {{
            font-size:20px;
            color:#dddddd;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )

set_bg()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload", "Result", "Dashboard"])

# ---------------- HOME ----------------
if page == "Home":

    st.markdown("""
    <div class="glass-container">
        <h1>🧠 Early Parkinson's Disease Detection System</h1>
        <p>AI-based analysis using spiral and wave drawings.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 🚀 Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("images/brain.png", width=150)
        st.markdown("### Brain Pattern Analysis")
        st.write("Analyze spiral and wave drawing patterns to identify early neurological abnormalities associated with Parkinson’s Disease.")

    with col2:
        st.image("images/ml.png", width=150)
        st.markdown("### AI Model")
        st.write("Advanced AI algorithms trained on medical datasets.")

    with col3:
        st.image("images/report.png", width=150)
        st.markdown("### Instant Prediction")
        st.write("Get results with risk classification.")

# ---------------- UPLOAD ----------------
elif page == "Upload":

    st.title("Upload Drawing Images")

    name, age, gender = get_user_details()
    spiral_path, wave_path = upload_images()

    st.markdown("---")

    if st.button("Generate Prediction"):

        errors = []

        if name.strip() == "":
            errors.append("Name is required.")

        if age < 10 or age > 100:
            errors.append("Enter valid age between 10-100.")

        if spiral_path is None:
            errors.append("Upload Spiral Image")

        if wave_path is None:
            errors.append("Upload Wave Image")

        if errors:
            for e in errors:
                st.warning(e)

        else:

            result = predict_parkinson(spiral_path, wave_path)

            if result is None:
                st.error("Prediction failed because models could not load.")
            else:
                st.session_state["result"] = result
                st.session_state["name"] = name
                st.session_state["age"] = age
                st.session_state["gender"] = gender

                st.success("Prediction Generated. Go to Result Page.")

# ---------------- RESULT ----------------
elif page == "Result":

    st.title("Patient Result")

    if "result" not in st.session_state:
        st.warning("Generate prediction first.")
    else:

        result = st.session_state["result"]

        name = st.session_state["name"]
        age = st.session_state["age"]
        gender = st.session_state["gender"]

        spiral_prob = result["spiral_prob"]
        wave_prob = result["wave_prob"]
        final_prob = result["final_prob"]
        diagnosis = result["diagnosis"]
        risk = result["risk_level"]
        spiral_heatmap = result.get("spiral_heatmap")
        wave_heatmap = result.get("wave_heatmap")

        st.subheader("Patient Details")

        st.write("Name:", name)
        st.write("Age:", age)
        st.write("Gender:", gender)

        st.markdown("---")

        st.subheader("Drawing Analysis")

        st.write(f"Spiral Probability: {spiral_prob*100:.0f}%")
        st.write(f"Wave Probability: {wave_prob*100:.0f}%")

        st.markdown("---")

        st.subheader("Combined AI Prediction")

        st.write(f"{final_prob*100:.0f}%")

        st.markdown("---")

        st.subheader("Final Diagnosis")

        if diagnosis == "Parkinson":
            st.error("Parkinson Detected")
        else:
            st.success("Healthy")
            st.success("The analysis indicates stable motor control patterns. Maintain a healthy lifestyle and regular neurological wellness monitoring.")

        st.markdown("---")

        st.subheader("Risk Level")

        if risk == "High":

            st.error("High Risk")

            st.warning(
                "The drawing patterns show strong tremor indicators associated with Parkinson’s Disease. "
                "It is strongly recommended to consult a neurologist or medical specialist for further clinical evaluation."
            )

        elif risk == "Medium":

            st.warning("Medium Risk")

            st.info(
                "Some tremor patterns were detected in the drawings. "
                "It is advisable to monitor symptoms and consider consulting a neurologist for professional assessment."
            )

        else:

            st.success("Low Risk")

            st.success(
                "The drawing patterns appear stable with minimal tremor indicators."
            )

        st.markdown("---")

        st.subheader("Explainable AI Heatmaps")

        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists(spiral_heatmap):
                st.image(spiral_heatmap, caption="Spiral Grad-CAM", use_container_width=True)
            else:
                st.warning("Spiral heatmap not generated")

        with col2:
            if os.path.exists(wave_heatmap):
                st.image(wave_heatmap, caption="Wave Grad-CAM", use_container_width=True)
            else:
                st.warning("Wave heatmap not generated")

        st.markdown("---")

        st.info("This system is intended for early screening support and does not replace professional medical diagnosis.")
# ---------------- DASHBOARD ----------------

elif page == "Dashboard":

    st.title("Patient Analysis Dashboard")

    if "result" not in st.session_state:
        st.warning("Generate prediction first.")
    else:

        result = st.session_state["result"]

        spiral_prob = result["spiral_prob"]
        wave_prob = result["wave_prob"]
        final_prob = result["final_prob"]
        diagnosis = result["diagnosis"]

        st.subheader("Model Scores")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Spiral Score", f"{spiral_prob*100:.0f}%")

        with col2:
            st.metric("Wave Score", f"{wave_prob*100:.0f}%")

        with col3:
            st.metric("Final Parkinson Probability", f"{final_prob*100:.0f}%")

        st.markdown("---")

        # Spiral vs Wave Comparison
        if diagnosis == "Healthy":
          st.subheader("Spiral vs Wave Stability Analysis")
        else:
           st.subheader("Spiral vs Wave Tremor Analysis")

        tremor_data = pd.DataFrame({
            "Drawing Type": ["Spiral", "Wave"],
            "Tremor Score": [spiral_prob*100, wave_prob*100]
        })

        fig1 = px.bar(
            tremor_data,
            x="Drawing Type",
            y="Tremor Score",
            color="Drawing Type",
            title="Drawing Tremor Comparison"
        )

        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")

        # HEALTHY CASE
        if diagnosis == "Healthy":

            st.subheader("Movement Stability Analysis")

            stable = 100 - final_prob*100

            stability_data = pd.DataFrame({
                "Movement Type": ["Stable Movement", "Minor Tremor"],
                "Percentage": [stable, 100-stable]
            })

            fig2 = px.bar(
                stability_data,
                x="Percentage",
                y="Movement Type",
                orientation="h",
                color="Movement Type",
                title="Drawing Stability"
            )

            st.plotly_chart(fig2, use_container_width=True)

            st.success("Patient shows stable drawing control patterns.")

        # PARKINSON CASE
        else:

            st.subheader("Tremor Severity Analysis")

            severity_data = pd.DataFrame({
                "Factor": [
                    "Overall Tremor Severity",
                    "Spiral Drawing Instability",
                    "Wave Drawing Instability"
                ],
                "Score": [
                    final_prob*100,
                    spiral_prob*100,
                    wave_prob*100
                ]
            })

            fig3 = px.bar(
                severity_data,
                x="Factor",
                y="Score",
                color="Factor",
                title="Motor Instability Indicators"
            )

            st.plotly_chart(fig3, use_container_width=True)

            st.error("Drawing patterns show tremor characteristics associated with Parkinson's Disease.")