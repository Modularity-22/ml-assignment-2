"""
ML Assignment 2 - Streamlit App
Student: Mridul Gupta | ID: 2026AJ05726
Dataset: Heart Disease (Cleveland UCI)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
import io

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease ML Classifier",
    page_icon="❤️",
    layout="wide"
)

# ── Constants ──────────────────────────────────────────────────────
MODEL_FILES = {
    'Logistic Regression': 'model/logistic_regression.joblib',
    'Decision Tree': 'model/decision_tree.joblib',
    'KNN': 'model/knn.joblib',
    'Naive Bayes': 'model/naive_bayes.joblib',
    'Support Vector Machine': 'model/svm.joblib'
}

FEATURE_DESCRIPTIONS = {
    'age': 'Age (years)',
    'sex': 'Sex (1=Male, 0=Female)',
    'cp': 'Chest Pain Type (0-3)',
    'trestbps': 'Resting Blood Pressure (mmHg)',
    'chol': 'Serum Cholesterol (mg/dl)',
    'fbs': 'Fasting Blood Sugar > 120mg/dl (1=True)',
    'restecg': 'Resting ECG Results (0-2)',
    'thalach': 'Max Heart Rate Achieved',
    'exang': 'Exercise Induced Angina (1=Yes)',
    'oldpeak': 'ST Depression induced by exercise',
    'slope': 'Slope of peak exercise ST segment (0-2)',
    'ca': 'Number of major vessels colored by fluoroscopy (0-4)',
    'thal': 'Thalassemia (0=Normal, 1=Fixed Defect, 2=Reversible Defect, 3=Unknown)',
    'target': 'Target (1=Heart Disease, 0=No Disease)'
}

MODEL_OBSERVATIONS = {
    'Logistic Regression': (
        "Logistic Regression provides a solid baseline with good interpretability. "
        "It performs well on linearly separable features but may underfit complex non-linear patterns in the data. "
        "Feature scaling helps it converge effectively."
    ),
    'Decision Tree': (
        "Decision Tree captures non-linear relationships and is easy to interpret. "
        "With max_depth=5, it balances bias-variance tradeoff. It is prone to overfitting on noisy data "
        "but provides human-readable decision rules, which is valuable in medical contexts."
    ),
    'KNN': (
        "KNN (k=7) is a lazy learner that classifies based on proximity. "
        "It performs well when the data has clear local structure but can be slow on large datasets. "
        "Feature scaling is essential for KNN to work correctly."
    ),
    'Naive Bayes': (
        "Gaussian Naive Bayes assumes feature independence, which rarely holds in medical data. "
        "Despite this assumption, it often performs surprisingly well and is very fast to train. "
        "It may underperform on correlated features like cholesterol and trestbps."
    ),
    'Support Vector Machine': (
        "SVM with RBF kernel excels at finding optimal decision boundaries in high-dimensional space. "
        "It typically achieves the highest accuracy on this dataset by maximizing the margin. "
        "It is robust to outliers but computationally intensive for large datasets."
    )
}


@st.cache_resource
def load_models():
    """Load all saved models and scaler."""
    models = {}
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    scaler = joblib.load('model/scaler.joblib') if os.path.exists('model/scaler.joblib') else None
    feature_names = joblib.load('model/feature_names.joblib') if os.path.exists('model/feature_names.joblib') else None
    return models, scaler, feature_names


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute all required evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = roc_auc_score(y_true, y_pred)
    else:
        auc = roc_auc_score(y_true, y_pred)
    return {
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1 Score': round(f1, 4),
        'MCC': round(mcc, 4)
    }


def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Disease', 'Heart Disease'])
    ax.set_yticklabels(['No Disease', 'Heart Disease'])
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=12, fontweight='bold')
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    return fig


def plot_metrics_bar(metrics_dict):
    """Bar chart comparing metrics across models."""
    df = pd.DataFrame(metrics_dict).T
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df.index))
    width = 0.13
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
    metric_cols = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
    for i, (metric, color) in enumerate(zip(metric_cols, colors)):
        if metric in df.columns:
            vals = df[metric].values
            bars = ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Comparison — Evaluation Metrics', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(df.index, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


# ── MAIN APP ───────────────────────────────────────────────────────
def main():
    # Header
    st.title("❤️ Heart Disease Classification — ML Assignment 2")
    st.markdown(
        """
        **Student:** Mridul Gupta &nbsp;|&nbsp; **ID:** 2026AJ05726 &nbsp;|&nbsp;
        **Dataset:** Heart Disease UCI (Cleveland) &nbsp;|&nbsp; **Task:** Binary Classification
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.markdown("### Step 1: Upload Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload test_data.csv", type=['csv'],
        help="Upload the test CSV containing all 13 features + target column"
    )

    st.sidebar.markdown("### Step 2: Select Model")
    selected_model_name = st.sidebar.selectbox(
        "Choose Classifier",
        list(MODEL_FILES.keys()),
        index=4  # Default: SVM
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Dataset Info**")
    st.sidebar.info(
        "**Heart Disease (Cleveland)**\n\n"
        "- Source: UCI ML Repository\n"
        "- Features: 13\n"
        "- Instances: 303\n"
        "- Task: Binary Classification\n"
        "- Target: 1 = Heart Disease\n"
        "         0 = No Disease"
    )

    # Load models
    models, scaler, feature_names = load_models()

    if not models:
        st.error(
            "⚠️ No trained models found in `model/` folder.\n\n"
            "Please run `train_models.py` first to train and save all 5 classifiers."
        )
        return

    # ── Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Model Evaluation", "📈 All Models Comparison", "📋 Dataset Info", "ℹ️ About"]
    )

    # ── TAB 1: Single model evaluation ──────────────────────────
    with tab1:
        st.subheader(f"📊 {selected_model_name} — Evaluation on Test Data")

        if uploaded_file is None:
            st.info("👈 Upload `test_data.csv` from the sidebar to begin evaluation.")
            with st.expander("Expected CSV format"):
                st.markdown("""
                The CSV must contain these columns:
                ```
                age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal, target
                ```
                - All 13 feature columns are required
                - `target` column (0 or 1) is required for metric calculation
                """)
            return

        try:
            test_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        st.write(f"**Uploaded data shape:** {test_df.shape}")

        # Validate columns
        required_cols = feature_names + ['target'] if feature_names else []
        missing = [c for c in required_cols if c not in test_df.columns]
        if missing:
            st.error(f"Missing columns in uploaded file: {missing}")
            return

        X_test = test_df[feature_names]
        y_test = test_df['target']

        # Scale
        if scaler:
            X_test_sc = scaler.transform(X_test)
        else:
            X_test_sc = X_test.values

        # Predict
        model = models[selected_model_name]
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1] if hasattr(model, 'predict_proba') else None

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)

        # Display metrics in colored cards
        st.markdown("### 📐 Evaluation Metrics")
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("🎯 Accuracy", f"{metrics['Accuracy']:.4f}")
        col2.metric("📉 AUC Score", f"{metrics['AUC']:.4f}")
        col3.metric("🔍 Precision", f"{metrics['Precision']:.4f}")
        col4.metric("📢 Recall", f"{metrics['Recall']:.4f}")
        col5.metric("⚖️ F1 Score", f"{metrics['F1 Score']:.4f}")
        col6.metric("🧮 MCC", f"{metrics['MCC']:.4f}")

        # Confusion matrix
        st.markdown("### 🗂️ Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = plot_confusion_matrix(cm, selected_model_name)
        col_cm, col_rep = st.columns([1, 1])
        with col_cm:
            st.pyplot(fig_cm)
        with col_rep:
            st.markdown("**Classification Report**")
            report = classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease'])
            st.text(report)

        # Model observations
        st.markdown("### 💡 Model Observation")
        st.info(MODEL_OBSERVATIONS.get(selected_model_name, "N/A"))

        # Raw predictions
        with st.expander("🔍 View Predictions vs Actual"):
            pred_df = X_test.copy()
            pred_df['Actual'] = y_test.values
            pred_df['Predicted'] = y_pred
            pred_df['Correct'] = pred_df['Actual'] == pred_df['Predicted']
            st.dataframe(pred_df, use_container_width=True)

    # ── TAB 2: All models comparison ────────────────────────────
    with tab2:
        st.subheader("📈 All 5 Models — Side-by-Side Comparison")

        if uploaded_file is None:
            st.info("👈 Upload test data from the sidebar to compare all models.")
            return

        # Re-read (streamlit file uploader can be seeked)
        uploaded_file.seek(0)
        test_df2 = pd.read_csv(uploaded_file)
        X_test2 = test_df2[feature_names]
        y_test2 = test_df2['target']
        X_test2_sc = scaler.transform(X_test2) if scaler else X_test2.values

        all_metrics = {}
        for name, model in models.items():
            y_pred2 = model.predict(X_test2_sc)
            y_prob2 = model.predict_proba(X_test2_sc)[:, 1] if hasattr(model, 'predict_proba') else None
            all_metrics[name] = compute_metrics(y_test2, y_pred2, y_prob2)

        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df.index.name = 'ML Model'

        st.markdown("### 📊 Comparison Table")
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, color='#90EE90').format('{:.4f}'),
            use_container_width=True
        )

        st.markdown("### 📊 Metrics Bar Chart")
        fig_bar = plot_metrics_bar(all_metrics)
        st.pyplot(fig_bar)

        # Find winner
        winner = metrics_df['F1 Score'].idxmax()
        st.success(f"🏆 **Best Model (by F1 Score):** {winner}")

        # Observations table
        st.markdown("### 💡 Model Observations")
        obs_data = {
            'ML Model': list(MODEL_OBSERVATIONS.keys()),
            'Observation': list(MODEL_OBSERVATIONS.values())
        }
        obs_df = pd.DataFrame(obs_data).set_index('ML Model')
        st.table(obs_df)

    # ── TAB 3: Dataset info ──────────────────────────────────────
    with tab3:
        st.subheader("📋 Dataset Information")
        st.markdown("""
        ### Heart Disease (Cleveland) Dataset

        **Source:** UCI Machine Learning Repository  
        **URL:** https://archive.ics.uci.edu/ml/datasets/Heart+Disease  
        **Problem Type:** Binary Classification  
        **Instances:** 303  
        **Features:** 13  
        **Target:** Presence of Heart Disease (0 = No, 1 = Yes)

        ---

        ### Feature Descriptions
        """)
        feat_df = pd.DataFrame(
            [(k, v) for k, v in FEATURE_DESCRIPTIONS.items()],
            columns=['Feature', 'Description']
        )
        st.table(feat_df)

        st.markdown("""
        ---
        ### Preprocessing Steps
        1. Missing values (`?`) dropped
        2. Target binarized: 0 = No Disease, 1 = Disease (original had values 0-4)
        3. Features standardized using `StandardScaler`
        4. 80/20 train-test split with stratification
        """)

    # ── TAB 4: About ─────────────────────────────────────────────
    with tab4:
        st.subheader("ℹ️ About This App")
        st.markdown("""
        ### ML Assignment 2 — BITS WILP M.Tech (AIML)
        **Course:** Machine Learning  
        **Student:** Mridul Gupta  
        **Student ID:** 2026AJ05726  
        **Submission Deadline:** 08-Nov-2026

        ---

        ### Models Implemented
        | Model | Library | Key Hyperparameter |
        |-------|---------|-------------------|
        | Logistic Regression | scikit-learn | max_iter=1000 |
        | Decision Tree | scikit-learn | max_depth=5 |
        | K-Nearest Neighbor | scikit-learn | n_neighbors=7 |
        | Naive Bayes | scikit-learn | GaussianNB |
        | Support Vector Machine | scikit-learn | kernel=rbf |

        ---

        ### How to Use
        1. Upload `test_data.csv` from the sidebar
        2. Select a model from the dropdown
        3. View metrics and confusion matrix in **Model Evaluation** tab
        4. Compare all models in **All Models Comparison** tab

        ---

        ### Project Structure
        ```
        ml_assignment2/
        ├── app.py                   # This Streamlit app
        ├── train_models.py          # Training script
        ├── requirements.txt         # Dependencies
        ├── README.md                # Documentation
        ├── test_data.csv            # Test dataset
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
        """)


if __name__ == '__main__':
    main()
