import streamlit as st
import joblib

model = joblib.load("models/startup_model.pkl")

st.title("AI Startup Success Predictor")

funding_rounds = st.number_input("Funding Rounds", min_value=0)
milestones = st.number_input("Milestones", min_value=0)
relationships = st.number_input("Relationships", min_value=0)

if st.button("Predict"):
    probability = model.predict_proba([[funding_rounds, milestones, relationships]])
    success_rate = probability[0][1] * 100

    st.write(f"Success Probability: {success_rate:.2f}%")

    if success_rate >= 50:
        st.success("Startup likely to succeed 🚀")
    else:
        st.error("Startup may struggle ⚠️")