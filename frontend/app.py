import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("models/student_pass_predictor.pkl")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3 {
        color: #00c0ff;
    }
    .stButton>button {
        background-color: #00c0ff;
        color: white;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Visuals", "ℹ️ About"])

# --- Tab 1: Prediction ---
with tab1:
    st.header("🎓 Student Pass Predictor")

    age = st.number_input("Age", min_value=15, max_value=25, value=18)
    studytime = st.selectbox("Study Time (1=low, 4=high)", [1, 2, 3, 4])
    failures = st.number_input("Past Failures", min_value=0, max_value=5, value=0)
    absences = st.number_input("Absences", min_value=0, max_value=50, value=5)
    G1 = st.number_input("First Period Grade (G1)", min_value=0, max_value=20, value=12)
    G2 = st.number_input("Second Period Grade (G2)", min_value=0, max_value=20, value=14)

    if st.button("Predict"):
        sample = pd.DataFrame([[age, studytime, failures, absences, G1, G2]],
                              columns=["age", "studytime", "failures", "absences", "G1", "G2"])
        prediction = model.predict(sample)
        result = "✅ Pass" if prediction[0] == 1 else "❌ Fail"
        st.success(f"Prediction: {result}")

# --- Tab 2: Visuals ---
with tab2:
    st.header("📈 Model Visuals")

    # Example: Feature Importance
    importance = model.feature_importances_
    features = ["age", "studytime", "failures", "absences", "G1", "G2"]
    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=features, ax=ax, palette="cool")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# --- Tab 3: About ---
with tab3:
    st.header("ℹ️ About Project")
    st.write("""
    CareerFitPipeline_Phase1 is a complete ML pipeline project.
    - Dataset: Student performance (Math + Portuguese)
    - Model: Random Forest (Tuned)
    - Accuracy: ~91%
    - Dashboard: Streamlit + Custom CSS
    """)
