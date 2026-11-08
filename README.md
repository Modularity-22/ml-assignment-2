# ❤️ Heart Disease ML Classifier — ML Assignment 2

**Student:** Mridul Gupta  
**Student ID:** 2026AJ05726  
**Course:** Machine Learning — M.Tech (AIML), BITS WILP  
**Submission Date:** 08-Nov-2026  
**Submission Date:** 08-Nov-2026

---

## a. Problem Statement

This project implements and compares **5 machine learning classification models** to predict the presence of **Heart Disease** in patients based on clinical features. The goal is to build an end-to-end ML pipeline — from data preprocessing to model deployment as an interactive web app.

---

## b. Dataset Description

| Property | Details |
|----------|---------|
| **Name** | Heart Disease (Cleveland) |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/ml/datasets/Heart+Disease |
| **Problem Type** | Binary Classification |
| **Instances** | 303 |
| **Features** | 13 |
| **Target** | 0 = No Disease, 1 = Heart Disease |
| Property         | Details                                               |
| ---------------- | ----------------------------------------------------- |
| **Name**         | Heart Disease (Cleveland)                             |
| **Source**       | UCI Machine Learning Repository                       |
| **URL**          | https://archive.ics.uci.edu/ml/datasets/Heart+Disease |
| **Problem Type** | Binary Classification                                 |
| **Instances**    | 303                                                   |
| **Features**     | 13                                                    |
| **Target**       | 0 = No Disease, 1 = Heart Disease                     |

### Feature Descriptions

| Feature | Description |
|---------|-------------|
| `age` | Age of the patient (years) |
| `sex` | Sex (1=Male, 0=Female) |
| `cp` | Chest Pain Type (0–3) |
| `trestbps` | Resting Blood Pressure (mmHg) |
| `chol` | Serum Cholesterol (mg/dl) |
| `fbs` | Fasting Blood Sugar > 120 mg/dl (1=True, 0=False) |
| `restecg` | Resting ECG Results (0–2) |
| `thalach` | Maximum Heart Rate Achieved |
| `exang` | Exercise Induced Angina (1=Yes, 0=No) |
| `oldpeak` | ST Depression Induced by Exercise |
| `slope` | Slope of Peak Exercise ST Segment (0–2) |
| `ca` | Number of Major Vessels Coloured by Fluoroscopy (0–4) |
| `thal` | Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect) |
| Feature    | Description                                                 |
| ---------- | ----------------------------------------------------------- |
| `age`      | Age of the patient (years)                                  |
| `sex`      | Sex (1=Male, 0=Female)                                      |
| `cp`       | Chest Pain Type (0–3)                                       |
| `trestbps` | Resting Blood Pressure (mmHg)                               |
| `chol`     | Serum Cholesterol (mg/dl)                                   |
| `fbs`      | Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)           |
| `restecg`  | Resting ECG Results (0–2)                                   |
| `thalach`  | Maximum Heart Rate Achieved                                 |
| `exang`    | Exercise Induced Angina (1=Yes, 0=No)                       |
| `oldpeak`  | ST Depression Induced by Exercise                           |
| `slope`    | Slope of Peak Exercise ST Segment (0–2)                     |
| `ca`       | Number of Major Vessels Coloured by Fluoroscopy (0–4)       |
| `thal`     | Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect) |

### Preprocessing

- Missing values (`?`) removed (dropped rows)
- Target binarized: values ≥ 1 → 1 (Disease), 0 → 0 (No Disease)
- Features scaled with `StandardScaler`
- 80/20 train-test split with stratification (`random_state=42`)

---

## c. GitHub Repository Link

🔗 **[https://github.com/YOUR_USERNAME/ml-assignment-2](https://github.com/YOUR_USERNAME/ml-assignment-2)**

Repository contains:

- `app.py` — Streamlit web application
- `train_models.py` — Model training script
- `requirements.txt` — Python dependencies
- `README.md` — This documentation
- `test_data.csv` — Test dataset (48 samples)
- `model/` — Saved trained models (`.joblib` files)

---

## d. Models Used & Evaluation Metrics

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8750 | 0.9411 | 0.8889 | 0.8000 | 0.8421 | 0.7419 |
| Decision Tree | 0.8333 | 0.8375 | 0.8333 | 0.7500 | 0.7895 | 0.6547 |
| KNN | 0.7708 | 0.8804 | 0.7368 | 0.7000 | 0.7179 | 0.5257 |
| Naive Bayes | **0.9167** | 0.9071 | **0.9444** | **0.8500** | **0.8947** | **0.8292** |
| Support Vector Machine | 0.8750 | 0.8696 | 0.8889 | 0.8000 | 0.8421 | 0.7419 |
| ML Model Name          | Accuracy   | AUC    | Precision  | Recall     | F1         | MCC        |
| ---------------------- | ---------- | ------ | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression    | 0.8750     | 0.9411 | 0.8889     | 0.8000     | 0.8421     | 0.7419     |
| Decision Tree          | 0.8333     | 0.8375 | 0.8333     | 0.7500     | 0.7895     | 0.6547     |
| KNN                    | 0.7708     | 0.8804 | 0.7368     | 0.7000     | 0.7179     | 0.5257     |
| Naive Bayes            | **0.9167** | 0.9071 | **0.9444** | **0.8500** | **0.8947** | **0.8292** |
| Support Vector Machine | 0.8750     | 0.8696 | 0.8889     | 0.8000     | 0.8421     | 0.7419     |

> **Bold** = Best score for that metric

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|--------------------------------------|
| Logistic Regression | Delivers a strong baseline with high AUC (0.94), indicating excellent discriminative ability. Its linear decision boundary works well because many features (like `cp`, `thal`, `ca`) have near-linear separation. Interpretable via coefficients. Fast to train and deploy. |
| Decision Tree | Performs moderately well (Acc: 0.83). With max_depth=5, it captures some non-linear interactions but may miss complex patterns. Its interpretability via decision rules is valuable in a clinical setting. Slightly lower AUC suggests less probabilistic calibration. |
| KNN | Lowest accuracy (0.77) among all models on this dataset. KNN is sensitive to the curse of dimensionality with 13 features. Performance improves with scaling (applied here), but still struggles with the spread of the feature space. Best suited for smaller, denser datasets. |
| Naive Bayes | Surprisingly the **best performer** (Acc: 0.9167, MCC: 0.8292) despite its feature independence assumption. The features appear sufficiently independent after scaling, and GaussianNB's probabilistic approach aligns well with the medical nature of the data. Fast and robust. |
| Support Vector Machine | Ties Logistic Regression on Accuracy and F1. The RBF kernel effectively handles non-linearity but its AUC (0.87) is lower, suggesting its probability calibration is less refined than LR (Platt scaling may need tuning). Strong overall and highly generalizable. |
| **Overall Winner** | 🏆 **Naive Bayes** — Best accuracy (91.67%), highest MCC (0.83), best precision (94.44%) and recall (85%), and best F1 score (89.47%). Despite its simplicity, Gaussian Naive Bayes is the top performer on this medical dataset, likely due to the near-independence of standardized clinical features. |
| ML Model Name          | Observation about Model Performance                                                                                                                                                                                                                                                                      |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression    | Delivers a strong baseline with high AUC (0.94), indicating excellent discriminative ability. Its linear decision boundary works well because many features (like `cp`, `thal`, `ca`) have near-linear separation. Interpretable via coefficients. Fast to train and deploy.                             |
| Decision Tree          | Performs moderately well (Acc: 0.83). With max_depth=5, it captures some non-linear interactions but may miss complex patterns. Its interpretability via decision rules is valuable in a clinical setting. Slightly lower AUC suggests less probabilistic calibration.                                   |
| KNN                    | Lowest accuracy (0.77) among all models on this dataset. KNN is sensitive to the curse of dimensionality with 13 features. Performance improves with scaling (applied here), but still struggles with the spread of the feature space. Best suited for smaller, denser datasets.                         |
| Naive Bayes            | Surprisingly the **best performer** (Acc: 0.9167, MCC: 0.8292) despite its feature independence assumption. The features appear sufficiently independent after scaling, and GaussianNB's probabilistic approach aligns well with the medical nature of the data. Fast and robust.                        |
| Support Vector Machine | Ties Logistic Regression on Accuracy and F1. The RBF kernel effectively handles non-linearity but its AUC (0.87) is lower, suggesting its probability calibration is less refined than LR (Platt scaling may need tuning). Strong overall and highly generalizable.                                      |
| **Overall Winner**     | 🏆 **Naive Bayes** — Best accuracy (91.67%), highest MCC (0.83), best precision (94.44%) and recall (85%), and best F1 score (89.47%). Despite its simplicity, Gaussian Naive Bayes is the top performer on this medical dataset, likely due to the near-independence of standardized clinical features. |

---

## Streamlit App Link

🌐 **[https://ml-assignment-2-f5wqt7ftmi6cckxhntharw.streamlit.app/](https://ml-assignment-2-f5wqt7ftmi6cckxhntharw.streamlit.app/)**

### App Features

- ✅ CSV upload for test data
- ✅ Model selection dropdown (5 classifiers)
- ✅ All 6 evaluation metrics displayed
- ✅ Confusion matrix visualization
- ✅ Classification report
- ✅ All-models comparison table with bar chart
- ✅ Model-wise observations

---

## Project Structure

```
ml-assignment-2/
├── app.py                      # Streamlit web application
├── train_models.py             # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── test_data.csv               # Test dataset (48 samples, 13 features + target)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── scaler.joblib
    ├── feature_names.joblib
    └── metrics_results.csv
```

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-assignment-2.git
cd ml-assignment-2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (generates model/ files and test_data.csv)
python train_models.py

# 4. Launch the Streamlit app
streamlit run app.py
```

---

*BITS Pilani, WILP Division — M.Tech (AIML) — Machine Learning — Assignment 2*
_BITS Pilani, WILP Division — M.Tech (AIML) — Machine Learning — Assignment 2_
