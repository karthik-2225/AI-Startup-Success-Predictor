import streamlit as st
st.markdown(
    """
    <meta name="google-site-verification" content="HQI8zpNuD3Elmlt3j2UmoIyG_LjdoR_Na5_QVrUL0OQ" />
    """,
    unsafe_allow_html=True
)
import joblib
st.set_page_config(
    page_title="AI Startup Success Predictor",
    page_icon="🚀",
    layout="centered"
)

model = joblib.load("models/startup_model.pkl")
st.title("🚀 AI Startup Success Predictor")
st.markdown("Predict the probability of startup success using Machine Learning.")

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
        st.markdown("---")
st.markdown(
    "<center><b>🚀 Developed by Karthik Enugula</b></center>",
    unsafe_allow_html=True
)
