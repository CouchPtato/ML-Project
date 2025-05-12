import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

print("📥 Loading dataset...")
df = pd.read_csv("heart.csv")
print("✅ Dataset loaded successfully!")
print(f"📊 Dataset shape: {df.shape}")
print(df.head())

label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # M -> 1, F -> 0
df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])  # Y -> 1, N -> 0

# One-hot encoding multi-class columns
df = pd.get_dummies(df, columns=['ChestPainType', 'ST_Slope'], drop_first=True)

df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])  # Normal->0, ST->1, LVH->2

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Train-test split
print("\n🔀 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"🧪 Train size: {X_train.shape[0]} | 🧾 Test size: {X_test.shape[0]}")

# Imputing missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print("\n🎯 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
print("✅ Model trained successfully!")

# Checking Accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy on test set: {accuracy * 100:.2f}%")

# === FUNCTION TO HANDLE PARTIAL OR FULL INPUT ===
def predict_heart_disease(user_input_dict):
    input_df = pd.DataFrame([user_input_dict])
    input_df = input_df.reindex(columns=X.columns, fill_value=np.nan)

    # Keep track of missing fields
    missing_fields = input_df.columns[input_df.isna().any()].tolist()

    # Impute missing values
    input_imputed = imputer.transform(input_df)

    # Scale the input
    input_scaled = scaler.transform(input_imputed)

    # Predict
    prediction = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[0][1] * 100  # probability of class 1

    if prediction[0] == 1:
        print(f"\n✅ Prediction: Person is likely to have heart disease. (Confidence: {proba:.2f}%)")
    else:
        print(f"\n❌ Prediction: Person is unlikely to have heart disease. (Confidence: {100 - proba:.2f}%)")

    if missing_fields:
        print(f"⚠️ Missing input features: {missing_fields}")
        print("⚠️ Note: Prediction may be less accurate due to missing data.")

# --- Partial Input ---
print("\n🔍 Predicting for partial input...")
partial_input = {
    "Age": 63,
    "Sex": 1,
    "RestingBP": 145,
    "FastingBS": 1,
    "RestingECG": 0,
    "ExerciseAngina": 0,
    "Oldpeak": 2.3,
    "ChestPainType_ATA": 1,
    "ChestPainType_NAP": 0,
    "ChestPainType_ASY": 0,
    "ST_Slope_Flat": 0,
    "ST_Slope_Up": 1
    # Cholesterol and MaxHR missing intentionally
}
predict_heart_disease(partial_input)

# --- Full Input ---
print("\n🔍 Predicting for full input...")

# For a patient with confirmed Heart problem to check accuracy.
full_input = {
    "Age": 65,
    "Sex": 1,  # Male
    "RestingBP": 170,  # High blood pressure
    "Cholesterol": 300,  # High cholesterol
    "FastingBS": 1,  # Fasting blood sugar > 120 mg/dl
    "RestingECG": 2,  # LVH - sign of heart strain
    "MaxHR": 110,  # Lower maximum heart rate
    "ExerciseAngina": 1,  # Yes - angina during exercise
    "Oldpeak": 3.5,  # High ST depression
    "ChestPainType_ATA": 0,
    "ChestPainType_NAP": 0,
    "ChestPainType_ASY": 1,  # Asymptomatic chest pain
    "ST_Slope_Flat": 1,  # Flat slope - riskier
    "ST_Slope_Up": 0
}

predict_heart_disease(full_input)
