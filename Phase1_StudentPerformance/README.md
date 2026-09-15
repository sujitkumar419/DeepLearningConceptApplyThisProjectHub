# 📊 Phase 1: Student Performance & Career Recommendation System

<p align="center">
  <img src="https://shields.io" alt="Python Version">
  <img src="https://shields.io" alt="Streamlit">
  <img src="https://shields.io" alt="Scikit-Learn">
  <img src="https://shields.io" alt="Pipeline Status">
</p>

## 📌 Overview
This repository hosts **Phase 1** of the **CareerFitPipeline**. This modules focuses on parsing multi-dimensional student academic data to predict pass/fail outcomes and dynamically recommend tailored career paths based on performance metrics. The goal is to establish a modular, production-ready machine learning workflow with automated data validation, robust feature pipelines, and an interactive decision-support interface.

---

## 🚀 Quick Start (Direct Run)

Execute the following commands in your terminal to deploy the dashboard locally:

```bash
# 1. Clone the master repository
git clone https://github.com

# 2. Navigate straight to the Phase 1 environment
cd DeepLearningConceptApplyThisProjectHub/Phase1_StudentPerformance

# 3. Install core dependencies securely
pip install -r requirements.txt

# 4. Launch the local interactive dashboard
streamlit run frontend/app.py
```

---

## 📂 Architecture & Directory Layout

```text
Phase1_StudentPerformance/
├── data/                # Raw and curated academic datasets
├── notebooks/           # Jupyter workspaces for Exploratory Data Analysis (EDA)
├── src/                 # Modular, production-grade source engines
│   ├── data_loader.py   # Automated ingestion & validation pipelines
│   ├── train.py         # Model training, hyperparameter optimization & serialization
│   ├── predict.py       # Low-latency inference engine
│   └── visualization.py # Analytical plotting scripts (Confusion Matrices, ROC)
├── models/              # Serialized artifact storage (.pkl / .joblib format)
├── frontend/            # Streamlit dashboard layout components
├── summary/             # Phase closure reports & technical briefs
├── requirements.txt     # Locked production dependencies
└── README.md            # Module system documentation
```

---

## 🛠️ Tech Stack & Dependencies

*   **Core Architecture:** `Python 3.9+`
*   **Data Processing:** `NumPy`, `Pandas`
*   **Machine Learning Ecosystem:** `Scikit-learn`
*   **Analytical Visualizations:** `Matplotlib`, `Seaborn`
*   **Interface Layer:** `Streamlit`

---

## 🧠 System Architecture & Workflow

1. **Ingestion (`src/data_loader.py`):** Loads operational student performance tracking matrices, handling missing data anomalies and encoding categorical data blocks.
2. **Feature Optimization:** Isolates academic variables, regularizes distributions, and scales metrics to mitigate feature bias.
3. **Inference & Strategy Evaluation:** Passes processed arrays into the prediction engine to determine risk flags (Pass/Fail) and map specific skill profiles to recommended career vectors.
4. **Presentation UI (`frontend/app.py`):** Serves analytical summaries and dynamic user prediction fields over a streamlined Streamlit web engine.

---

## 📈 Next Milestones (Pipeline Lifecycle)
*   [x] **Phase 1:** Student Performance Analytics & Decision Engine
*   [ ] **Phase 2:** Deep Learning Perceptron Infrastructure Construction
*   [ ] **Phase 3:** High-Dimensional Model Comparison (Tree-Ensembles vs. Linear Models)

---
<p align="center">Developed as part of the <b>CareerFitPipeline</b> Initiative.</p>
