import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# ----------------------------------------
# CONFIG
# ----------------------------------------
DATA_DIR = "dataset"        # Folder containing all CSVs
MODEL_DIR = "models"        # Folder where model will be saved
os.makedirs(MODEL_DIR, exist_ok=True)

X = []
y = []

print("🔍 Reading CSV files from:", DATA_DIR)

# ----------------------------------------
# LOAD ALL CSV FILES
# ----------------------------------------
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        label = file.replace(".csv", "")   # class name = filename
        filepath = os.path.join(DATA_DIR, file)

        print("➡ Loading:", file)

        df = pd.read_csv(filepath, header=None)

        for i in range(len(df)):
            X.append(df.iloc[i].values)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("\n📊 Total Samples Loaded:", len(X))
print("📌 Classes Found:", sorted(set(y)))

# ----------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, shuffle=True, stratify=y
)

# ----------------------------------------
# TRAIN MODEL
# ----------------------------------------
print("\n⏳ Training RandomForest Model...")
model = RandomForestClassifier(n_estimators=400, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"\n✅ Training Complete! Model Accuracy: {accuracy:.4f}")

# ----------------------------------------
# SAVE MODEL
# ----------------------------------------
model_path = os.path.join(MODEL_DIR, "sign_model.pkl")
pickle.dump(model, open(model_path, "wb"))

print("\n💾 Model saved at:", model_path)
print("🎉 You can now run: python main.py")
