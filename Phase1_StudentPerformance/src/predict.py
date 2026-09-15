import pandas as pd
import joblib

def load_model(path: str):
    """Load the trained model from pickle file"""
    return joblib.load(path)

def predict_student(data: dict):
    """
    Predict student pass/fail using trained model.
    Args:
        data (dict): Student features in dictionary form
    Returns:
        str: "Pass" or "Fail"
    """
    # Convert dict to DataFrame with correct columns
    sample = pd.DataFrame([data], columns=["age", "studytime", "failures", "absences", "G1", "G2"])
    
    # Load model
    model = load_model("models/student_pass_predictor.pkl")
    
    # Predict
    prediction = model.predict(sample)
    return "Pass" if prediction[0] == 1 else "Fail"

if __name__ == "__main__":
    # Example input
    student_data = {
        "age": 18,
        "studytime": 2,
        "failures": 0,
        "absences": 5,
        "G1": 12,
        "G2": 14
    }
    result = predict_student(student_data)
    print("Prediction:", result)
