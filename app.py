import streamlit as st
import joblib
import numpy as np

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Adenovirus Detection Tool",
    page_icon="🧬",
    layout="centered"
)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("decision_tree_model.joblib")

# ----------------------------
# Title & Description
# ----------------------------
st.markdown(
    """
    # 🧬 Adenovirus Detection Tool  

    This tool predicts the **likelihood of Adenovirus infection** based on patient symptoms.  
    It is designed to support **early detection and preventive care**, especially in areas with limited healthcare access.  
     
    Always consult a **qualified healthcare professional** for diagnosis and treatment.  
    """
)

st.markdown("---")

# ----------------------------
# Sidebar Information
# ----------------------------
st.sidebar.header("ℹ️ About")
st.sidebar.write(
    """
    - Trained using **Decision Tree Classifier**  
    - Dataset: Patient health parameters (5,434 records)  
    - Target: *Adenoviruses* (Yes/No)  
    """
)

# ----------------------------
# Input Features
# ----------------------------
st.markdown("### 📝 Enter Patient Symptoms")

col1, col2 = st.columns(2)

with col1:
    breathing_problem = st.selectbox("Breathing Problem", ["No", "Yes"])
    pink_eye = st.selectbox("Pink Eye", ["No", "Yes"])
    pneumonia = st.selectbox("Pneumonia", ["No", "Yes"])
    fever = st.selectbox("Fever", ["No", "Yes"])

with col2:
    acute_gastro = st.selectbox("Acute Gastroenteritis", ["No", "Yes"])
    dry_cough = st.selectbox("Dry Cough", ["No", "Yes"])
    sore_throat = st.selectbox("Sore Throat", ["No", "Yes"])
    bladder_infection = st.selectbox("Bladder Infection", ["No", "Yes"])

# ----------------------------
# Encode Inputs
# ----------------------------
def encode(val):
    return 1 if val == "Yes" else 0

features = np.array([[
    encode(breathing_problem),
    encode(pink_eye),
    encode(pneumonia),
    encode(fever),
    encode(acute_gastro),
    encode(dry_cough),
    encode(sore_throat),
    encode(bladder_infection)
]])

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.error(
            "⚠️ **High likelihood of Adenovirus infection detected!**\n\n"
            "👉 Please consult a **healthcare professional immediately**."
        )
    else:
        st.success(
            "✅ **Low likelihood of Adenovirus infection.**\n\n"
            "👍 Stay healthy, but if symptoms persist, consult a doctor for confirmation."
        )


# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    """
    ⚠️ **Disclaimer:**  
    This tool is a **machine learning–based prediction aid**.  
    It is **not a substitute for professional medical advice, diagnosis, or treatment**.  
    Always seek medical advice from a qualified healthcare provider.  
    """
)