import pandas as pd

def load_student_data(path: str) -> pd.DataFrame:
    """
    Load student dataset from CSV file.
    Args:
        path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(path)
    return df
