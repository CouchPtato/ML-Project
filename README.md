# 🫀 Heart Disease Prediction using Machine Learning

## 📌 Problem Statement

Cardiovascular diseases are the leading cause of death globally. Early detection of heart disease can help in timely medical intervention and save lives. This project aims to develop a machine learning model that predicts the likelihood of heart disease based on several medical attributes.

The system also supports **partial input prediction**, meaning the user doesn’t need to fill all features — the model uses statistical imputation for missing fields and provides a confidence score.

---

## 🔍 Overview

This project builds and evaluates a machine learning model to predict the presence of heart disease using the [Heart Disease UCI dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction). The model is trained using the Random Forest algorithm and includes:

* **Data preprocessing**

  * Label Encoding
  * One-Hot Encoding
  * Feature Scaling
  * Missing value handling using imputation
* **Model training and evaluation**
* **Dynamic prediction**

  * Accepts partial input and adjusts for missing features
  * Predicts presence of heart disease along with confidence score
  * Displays missing features to inform about reduced prediction accuracy

---

## 🧠 Machine Learning Model

* **Algorithm Used**: Random Forest Classifier
* **Validation Metric**: Accuracy
* **Input Type**: Structured tabular input (age, sex, cholesterol, etc.)
* **Output**: Binary prediction (1 = likely heart disease, 0 = unlikely) with confidence %

---

## 🗃️ Dataset

* **Source**: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
* **Format**: CSV file (`heart.csv`)
* **Target Variable**: `HeartDisease` (0 = No, 1 = Yes)
* **Features**:

  * Age
  * Sex
  * RestingBP
  * Cholesterol
  * FastingBS
  * RestingECG
  * MaxHR
  * ExerciseAngina
  * Oldpeak
  * ChestPainType (one-hot encoded)
  * ST\_Slope (one-hot encoded)

---

## 🛠️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn numpy
```

### 3. Run the script

Make sure the `heart.csv` file is in the same directory as the script.

```bash
python heart_disease_predictor.py
```

---

## 📈 Sample Output

```
✅ Prediction: Person is likely to have heart disease. (Confidence: 92.40%)

⚠️ Warning: Missing input features - ['Cholesterol']
⚠️ Prediction may be less accurate due to incomplete data.
```

---

## 📌 Future Improvements

* Deploy using Flask/Streamlit for user-friendly web UI
* Add visualization of feature importances
* Support for other classifiers (e.g., XGBoost, Logistic Regression)
* Real-time API for hospital systems

---

## 🤝 Contributions

Contributions are welcome! If you’d like to improve the prediction model, UI, or add more features, feel free to open a pull request.
