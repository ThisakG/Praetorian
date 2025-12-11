import pandas as pd # type: ignore # lib used for storing data in a table and exporting CSVs
from sklearn.model_selection import train_test_split # type: ignore # used for splitting datasets

# loading the dataset to the script
df = pd.read_csv("data/synthetic_logins.csv")

# 80% train / 20% test
train_df, test_df = train_test_split(
    df,
    test_size=0.2, # 20% of the dataset is now the test set
    shuffle=True,
    stratify=df["label"]  # keep class balance between normal and anomalous data
)

# saving both datasets
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

# terminal outputs 
print("Train/Test split completed:")
print("Train:", train_df.shape)
print("Test:", test_df.shape)
