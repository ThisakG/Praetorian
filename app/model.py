# Since the main goal is to detect abnormal behavior, Isolation forest algorithm is used for the model
# It works with mostly normal data with a few anomalies
# no need for labels
# the output has an anomaly_score, a predicted_label and a risk_score (0-100)

import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.ensemble import IsolationForest # type: ignore # used to train anomaly detection model
import joblib # type: ignore # used to save the model to disk
import os # used in numerical operations 

def train_model(train_path="data/train_preprocessed.csv", save_model_path="models/isolation_forest.pkl"):
    # loading the preprocessed training set
    df = pd.read_csv(train_path)

    # seperate features and labels
    X = df.drop(columns=["label"]) 
    y = df["label"]

    # initiating the model
    # 0.05 since 5% anomalies expected
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42) 

    print("Training model...")
    print(X.dtypes)
    print(X.head()) 
    model.fit(X)

    # saving the model
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    joblib.dump(model, save_model_path)
    print(f"model saved to {save_model_path}")

    return model, X, y

# calculating scores (anomaly and risk)
def score_dataset(model, X):
    # predicting the anomalies where -1 = anomaly and 1 = normal
    preds = model.predict(X)

    # initiating raw anomaly scores (lower = more anomalous)
    scores = model.decision_function(X)

    # converting the score to get a percentage (the higher the percentage, more the risk)
    risk = (scores.min()-scores) / (scores.min()-scores.max())
    risk = (risk * 100).round(2)

    return preds, scores, risk

# saving the scored dataset
def save_scored_dataset(df, preds, scores, risk, output_path):
    df["predicted_label"] = np.where(preds == -1, 1, 0)
    df["anomaly_score"] = scores
    df["risk_score"] = risk

    df.to_csv(output_path, index=False)
    print("Scored dataset saved to {output_path}")

if __name__ == "__main__":
    # defining paths
    TRAIN = "data/train_preprocessed.csv"
    TEST = "data/test_preprocessed.csv"

    # training the model
    model, X_train, y_train = train_model(TRAIN)

    # scoring the training data
    preds_train, scores_train, risk_train = score_dataset(model, X_train)
    df_train = pd.read_csv(TRAIN)
    save_scored_dataset(df_train,preds_train, scores_train, risk_train, "data/train_scored.csv")

    # scoring the test data
    df_test = pd.read_csv(TEST)
    X_test = df_test.drop(columns=["label"])
    preds_test, scores_test, risk_test = score_dataset(model, X_test)
    save_scored_dataset(df_test,preds_test, scores_test, risk_test, "data/test_scored.csv")

    print ("Model trained and scored datasets created.")

    # the .pkl model contains the trained Isolation Forest model object
    # the learned trees, thresholds, parameters, numpy arrays 
    # all are stored in a serialized binary format 