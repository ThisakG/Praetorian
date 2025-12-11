# Praetorian — Anomaly Detection & Analytics Dashboard

Praetorian is a machine-learning–powered anomaly detection system built using <b>Isolation Forest<b>, featuring an interactive Streamlit web interface, real-time scoring, visualization, model evaluatiob, and CSV exporting capabilities. 

This learning project / demo was creating in order to get a glimpse of how security teams can analyse user session data, derive anomaly scores, detect risky behaviour and visualize insights. 

Dashboard: http://192.168.1.15:8502 


# 🚀 Features

### 🔍 1. **Preprocessing Python Scripts**
- Cleans raw authentication logs  ()
- Extracts temporal features  
- One-hot encodes categorical fields  
- Converts IPv4 addresses into numeric vectors  
- Scales numerical features using MinMaxScaler  
- Ensures test data matches the model’s training schema  

### 🧠 2. **Machine Learning Model**
- Uses an Isolation Forest for anomaly detection  
- Supports:
  - Decision function scoring  
  - Prediction output (`Normal` / `Anomaly`)  
  - Threshold-based risk scoring  

### 📊 3. **Interactive Streamlit Dashboard**
- Upload raw CSV files or preprocessed CSV files  
- Automatic preprocessing detection  
- Visual anomaly tables  
- Score distribution plots  
- Highlighting of suspicious events  
- Exportable results  

---

# 📁 Project Structure
```
Praetorian/
| 
├── app/
│ ├── dashboard.py # Streamlit UI
│ ├── preprocess.py # Preprocessing functions (train + test)
│ └── utils.py # (Optional) Helper functions
│
├── data/
│ ├── train.csv
│ ├── test.csv
│ ├── train_preprocessed.csv
│ ├── test_preprocessed.csv
│ └── scaler.save # Saved MinMaxScaler
│
├── models/
│ └── anomaly_model.pkl # Trained Isolation Forest
│
├── notebooks/
│ └── training.ipynb # Model development experiments
│
├── plots/
│ └── charts.png # Saved visualizations (optional)
│
└── README.md
```

# 🏢 System Architecture Visualized
```
 ┌───────────────────────────────────────┐
 │           Synthetic Data              │
 │     (data/generate_synthetic.py)      │
 └──────────────────────┬────────────────┘
                        │
                        ▼
 ┌───────────────────────────────────────┐
 │          Preprocessing Layer          │
 │          (src/preprocess.py)          │
 │   - timestamp → hour                  │
 │   - one-hot encode device_type        │
 │   - IP address → 4 numerical features │
 │   - scaling (MinMaxScaler)            │
 └──────────────────────┬────────────────┘
                        │
                        ▼
 ┌───────────────────────────────────────┐
 │          Model Training               │
 │         (models/train_model.py)       │
 │      - Isolation Forest               │
 │      - Model saved as .joblib         │
 └──────────────────────┬────────────────┘
                        │
                        ▼
 ┌───────────────────────────────────────┐
 │       Streamlit Dashboard (app/)      │
 │    Upload CSV → Preprocess/Score →    │
 │        Visualize anomalies            │
 └───────────────────────────────────────┘
```
---

# 🔧 Tech stack

- Python 3.10
- Python libraries: pandas, numpy, scikit-learn (Isolation forest), matplotlib, seaborn, joblib, streamlit
- HTML (UI)

---

# Why Isolation Forest

Robust for high-dimensional log data

Unsupervised → no need for exact labels

Outputs anomaly score + prediction

Lightweight & production-ready

# 🧬 Synthetic Dataset Generation

Synthetic data simulates user login events across devices, locations, and times.

Fields Included:

- timestamp

- muser_id

- device_type

- ip

-login_success

- session_duration

- failed_attempts

- location

- label (normal/anomalous)

# How anomalies were simulated

- Random IPs with rare subnets

- High failed login attempts

- Odd login hours

- Suspicious session durations

- Unusual device/location combos

- Generated using:

- data/generate_synthetic.py


# Seperated the sythetic data by a train-test split

train.csv (80%)
test.csv  (20%)

# 🧼 Preprocessing Logic (src/preprocess.py)

Preprocessing is identical for training and test data, with consistent handling of categorical and numerical fields.

✔ Convert timestamp → hour: This captures temporal behaviour while removing timezone noise.

✔ One-hot encode device_type: Example:device_Windows, device_Linux, device_MacOS, device_Android, device_iOS

✔ Test-time logic auto-creates missing device columns so the model never breaks.

✔ Convert IP ("A.B.C.D") → ip_1 … ip_4: Treats each octet as an independent feature.

✔ Scale numerical columns: Using MinMaxScaler saved as scaler.save

✔ Numerical fields: login_success, failed_attempts, session_duration, hour, ip_1, ip_2, ip_3, ip_4

✔ Output

Two files:

train_preprocessed.csv
test_preprocessed.csv

# 🤖 Model Training (models/train_model.py)

The script:

✔ Loads preprocessed train data

✔ Fits Isolation Forest

✔ Saves trained model → models/isolation_forest.joblib

✔ Optionally evaluates using accuracy and confusion matrix

# Model outputs:

✔ decision_function(X) → anomaly score
✔ predict(X) → [-1 = anomaly, 1 = normal]

# 📊 Streamlit Dashboard (app/dashboard.py)

The heart of the project.

Features:

✔ Auto upload CSV (preprocessed)

✔ Auto-detect whether preprocessing is needed

✔ Score dataset using Isolation Forest

✔ Display: Top anomalies, Score distribution, Device breakdown, Failed login patterns, Suspicious hours

✔ Download results as CSV



# ▶️ How to Run the Project

1️⃣ Clone the repository
git clone https://github.com/<yourusername>/Praetorian.git
cd Praetorian

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Train the model (optional)
python models/train_model.py

5️⃣ Run the Streamlit dashboard
streamlit run app/dashboard.py


# 🧪 How the Whole Project Works Together

1. Synthetic data created → stored in /data/

2. Preprocessing performed → saves scaler + preprocessed datasets

3. Model trained → saved as .joblib

4. Dashboard loads model + scaler

→ accepts new data
→ preprocesses if needed
→ generates anomaly score + prediction
→ displays results interactively
