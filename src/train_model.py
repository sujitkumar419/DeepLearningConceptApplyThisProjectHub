import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.data_loader import load_student_data

def train_and_save_model():
    # Load dataset
    df = load_student_data("Data/student-mat.csv")

    # Features & target
    X = df[["age", "studytime", "failures", "absences", "G1", "G2"]]
    y = df["pass"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train tuned Random Forest
    model = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "models/student_pass_predictor.pkl")
    print("Model saved to models/student_pass_predictor.pkl")

if __name__ == "__main__":
    train_and_save_model()
