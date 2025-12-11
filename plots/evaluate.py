import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import joblib # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # type: ignore 

# -------------------------------
# Paths
MODEL_PATH = "models/isolation_forest.pkl"
TEST_DATA_PATH = "data/test_preprocessed.csv"
PLOT_DIR = "plots/"  # make sure this folder exists

def evaluate_model():
# Load model and test data
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    # -------------------------------
    # Predict using Isolation Forest
    preds = model.predict(X_test)  # 1 for normal, -1 for anomaly
    scores = -model.decision_function(X_test)  # anomaly scores

    # Convert predictions to 0 (normal) / 1 (anomalous)
    pred_labels = np.where(preds == -1, 1, 0)

    # -------------------------------
    # Detection metrics
    true_positives = np.sum((pred_labels == 1) & (y_test == 1))
    false_positives = np.sum((pred_labels == 1) & (y_test == 0))
    true_negatives = np.sum((pred_labels == 0) & (y_test == 0))
    false_negatives = np.sum((pred_labels == 0) & (y_test == 1))

    detection_rate = true_positives / (true_positives + false_negatives + 1e-6)
    false_positive_rate = false_positives / (false_positives + true_negatives + 1e-6)

    print(f"Detection rate: {detection_rate:.2f}")
    print(f"False positive rate: {false_positive_rate:.2f}")

    # -------------------------------
    # Confusion matrix
    cm = confusion_matrix(y_test, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomalous"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(PLOT_DIR + "confusion_matrix.png")
    plt.close()

    # -------------------------------
    # Anomaly score distribution
    plt.figure(figsize=(8,5))
    sns.histplot(scores, bins=50, kde=True, color="salmon")
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.savefig(PLOT_DIR + "anomaly_score_distribution.png")
    plt.close()

    # -------------------------------
    # Scatter plot of scores vs index
    plt.figure(figsize=(10,5))
    plt.scatter(range(len(scores)), scores, c=pred_labels, cmap="coolwarm", s=10)
    plt.title("Anomaly Scores by Sample")
    plt.xlabel("Sample Index")
    plt.ylabel("Anomaly Score")
    plt.colorbar(label="Predicted Label (0=Normal,1=Anomalous)")
    plt.savefig(PLOT_DIR + "scores_scatter.png")
    plt.close()

    print("Evaluation complete. Plots saved to 'plots/' folder.")
    
    return{
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate
    }


if __name__ == "__main__":
    evaluate_model()
