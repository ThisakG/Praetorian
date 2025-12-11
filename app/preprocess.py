import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore # used for scaling numeric data
import joblib  # type: ignore # used for saving/loading scaler

# preprocessing train dataset
def preprocess_train(csv_path, save_path=None, scaler_path="scaler.save"):

    df = pd.read_csv(csv_path)
    
    # Timestamp feature
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df = df.drop(columns=['timestamp'])
    
    # One-hot encode device_type
    df = pd.get_dummies(df, columns=['device_type'], prefix='device')
    
    # Convert IP to numeric
    ip_split = df['ip'].str.split('.', expand=True).astype(int)
    ip_split.columns = ['ip_1', 'ip_2', 'ip_3', 'ip_4']
    df = pd.concat([df.drop(columns=['ip']), ip_split], axis=1)
    
    # Separate labels
    y = df['label'].map({'normal':0, 'anomalous':1})
    X = df.drop(columns=['label', 'user_id', 'location'])
    
    # Fit scaler on train
    num_cols = ['login_success', 'failed_attempts', 'session_duration', 'hour', 'ip_1', 'ip_2', 'ip_3', 'ip_4']
    scaler = MinMaxScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    
    # Save preprocessed CSV
    if save_path:
        X.join(y.rename('label')).to_csv(save_path, index=False)
    
    return X, y

# preprocessing test dataset
def preprocess_test(input_data, scaler_path="scaler.save", save_path=None):

    # If a DataFrame is passed → use it directly
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()

    # If a file path (string) is passed → read CSV
    else:
        df = pd.read_csv(input_data)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df = df.drop(columns=['timestamp'])
    
    df = pd.get_dummies(df, columns=['device_type'], prefix='device')
    
    # Make sure test has same device columns as train and if some device_type columns missing in test, add them with 0
    expected_cols = ['device_Windows','device_Linux','device_MacOS','device_Android','device_iOS']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    
    ip_split = df['ip'].str.split('.', expand=True).astype(int)
    ip_split.columns = ['ip_1', 'ip_2', 'ip_3', 'ip_4']
    df = pd.concat([df.drop(columns=['ip']), ip_split], axis=1)
    
    y = df['label'].map({'normal':0, 'anomalous':1})

    X = df.drop(columns=['label', 'user_id', 'location'])

    # Convert device booleans to integers
    device_cols = [col for col in X.columns if col.startswith("device_")]
    X[device_cols] = X[device_cols].astype(int)
    
    # Load scaler and transform test
    scaler = joblib.load(scaler_path)
    num_cols = ['login_success', 'failed_attempts', 'session_duration', 'hour', 'ip_1', 'ip_2', 'ip_3', 'ip_4']
    X[num_cols] = scaler.transform(X[num_cols])
    
    # Save preprocessed CSV
    if save_path:
        X.join(y.rename('label')).to_csv(save_path, index=False)
    
    return X, y


if __name__ == "__main__":
    # Preprocess training set
    X_train, y_train = preprocess_train("data/train.csv", save_path="data/train_preprocessed.csv")
    # Preprocess test set using the same scaler
    X_test, y_test = preprocess_test("data/test.csv", save_path="data/test_preprocessed.csv")
    
    print("Preprocessing complete!")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
