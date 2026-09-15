# Phase 1 Summary

- Conducted EDA on student datasets (math + Portuguese).
- Preprocessed features (age, studytime, failures, absences, G1, G2).
- Trained multiple models:
  - Logistic Regression → 89.8%
  - Decision Tree → 89.8%
  - Random Forest (Tuned) → 91.1%
  - Gradient Boosting → 91.1%
- Selected Random Forest (Tuned) as final model.
- Saved trained model as `student_pass_predictor.pkl`.
- Built Streamlit dashboard for predictions.
