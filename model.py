import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# 1. Load dataset
print("Loading dataset...")
df = pd.read_csv("heart.csv")
print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"First 5 rows:\n{df.head()}")

# 2. Label Encoding for binary columns
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # M -> 1, F -> 0
df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])  # Y -> 1, N -> 0

# 3. One-Hot Encoding for multi-class columns
df = pd.get_dummies(df, columns=['ChestPainType', 'ST_Slope'], drop_first=True)

# 4. Label Encoding for other categorical columns
df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])  # 'Normal' -> 0, 'ST' -> 1, 'LVH' -> 2

# 5. Split features and label
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# 6. Train-test split
print("\nSplitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# 7. Feature scaling
print("\nScaling the features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled successfully!")

# 8. Imputer setup (for partial inputs later)
imputer = SimpleImputer(strategy="mean")
imputer.fit(X_train)

# 9. Train the model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training complete!")

# 10. Evaluate accuracy
print("\nEvaluating model on test data...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# --- 11. Predict on partial input ---
print("\n--- Predicting for New Input (with optional missing fields) ---")

# Example partial input: Some fields are missing
partial_input = {
    "Age": 63,
    "Sex": 1,
    "RestingBP": 145,
    # "Cholesterol": missing
    "FastingBS": 1,
    "RestingECG": 0,
    "MaxHR": 150,
    "ExerciseAngina": 0,
    "Oldpeak": 2.3,
    "ChestPainType_ATA": 1,
    "ChestPainType_NAP": 0,
    "ChestPainType_ASY": 0,
    "ST_Slope_Flat": 0,
    "ST_Slope_Up": 1
}

# Step 1: Convert to DataFrame
input_df = pd.DataFrame([partial_input])

# Step 2: Align with training features, insert NaN for missing fields
input_df = input_df.reindex(columns=X_train.columns, fill_value=np.nan)

# Step 3: Impute missing values
input_imputed = imputer.transform(input_df)

# Step 4: Scale
input_scaled = scaler.transform(input_imputed)

# Step 5: Predict
prediction = model.predict(input_scaled)
proba = model.predict_proba(input_scaled)[0]
confidence = np.max(proba) * 100

# Step 6: Output result
if prediction[0] == 1:
    print(f"✅ Prediction: Person is likely to have heart disease. (Confidence: {confidence:.2f}%)")
else:
    print(f"❌ Prediction: Person is unlikely to have heart disease. (Confidence: {confidence:.2f}%)")

# Step 7: Show missing features (if any)
missing_features = input_df.columns[input_df.isna().any()].tolist()
if missing_features:
    print(f"\n⚠️ Warning: Missing input features - {missing_features}")
    print("⚠️ Prediction may be less accurate due to incomplete data.")
