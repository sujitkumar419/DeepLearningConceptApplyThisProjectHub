import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("visuals/confusion_matrix.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    df = df.sort_values("importance", ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x="importance", y="feature", data=df)
    plt.title("Feature Importance")
    plt.savefig("visuals/feature_importance.png")
    plt.close()
