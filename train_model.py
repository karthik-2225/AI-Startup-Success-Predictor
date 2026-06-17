import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/objects.csv")

# Create target column
df["success"] = (df["funding_total_usd"] > 1000000).astype(int)

# Features
X = df[[
    "funding_rounds",
    "milestones",
    "relationships"
]]

# Target
y = df["success"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
joblib.dump(model,"models/startup_model.pkl")
print("Model Saved Successfully ")