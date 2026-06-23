# Stroke Prediction — Hybrid Unsupervised + Supervised Machine Learning

A machine learning project that combines **unsupervised clustering** and **supervised classification** to predict stroke risk and uncover hidden patterns in patient health data.

> University of Jeddah — College of Computer Science and Engineering
> Machine Learning Course Project — Prepared for Dr. Elham Alghamdi

## 📄 Full Report

The complete written report — including literature review, methodology, and detailed discussion — is available here: [`Stroke_Prediction_Report.pdf`](Stroke_Prediction_Report.pdf)

## Problem

Stroke is the second leading cause of death globally, responsible for **7.25 million deaths in 2021 alone** (10.7% of all deaths worldwide), with global incidence up over 70% since 1990. Early identification of high-risk individuals can enable timely intervention and significantly improve outcomes — but traditional risk assessment tools often miss complex interactions between health factors.

## Approach

This project uses a **two-stage hybrid pipeline**:

1. **Unsupervised clustering** (without using the stroke label) to uncover latent high-risk / low-risk patient groups based on feature similarity
2. **Supervised classification**, using the cluster assignment as an additional engineered feature, to predict stroke occurrence

This addresses a common gap in prior research — most stroke prediction studies use raw features directly without deriving representations that reveal hidden structure, and many don't adequately handle class imbalance or multicollinearity.

## Dataset

**Stroke Prediction Dataset** ([Kaggle, fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)) — 5,110 patient records with features including age, gender, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, and smoking status. Target variable: `stroke` (1 = stroke occurred, 0 = no stroke).

**Class imbalance:** 4,861 non-stroke cases vs. 249 stroke cases — addressed using SMOTE.

To run the notebook yourself, download the CSV from Kaggle and save it in this folder as `our_data.csv`.

## Methodology

### Pre-processing
- Imputed missing `bmi` values with the mean; filled missing `smoking_status` with the mode
- Replaced the rare `Other` gender category with `Female`
- Label encoding for binary features (`gender`, `ever_married`, `Residence_type`); one-hot encoding for multi-class features (`work_type`, `smoking_status`)
- `StandardScaler` applied to numerical features (`age`, `avg_glucose_level`, `bmi`)
- **SMOTE** applied to balance the minority (stroke) class
- **PCA** applied for dimensionality reduction and 2D visualization
- 80/20 train-test split

### Unsupervised learning — clustering
Four models were trained and compared to identify natural high-risk / low-risk groupings:

| Model | Silhouette Score | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|
| **K-Means** | 0.774 | 446,467 | 0.313 |
| Hierarchical | 0.771 | 51,904 | 0.317 |
| GMM | 0.765 | 416,877 | 0.324 |
| DBSCAN | -0.077 | 58 | 1.266 |

**K-Means** was selected as the best-performing model and used to generate a risk-cluster feature for the supervised stage.

### Supervised learning — classification
Three classifiers were trained using the original features plus the K-Means cluster label:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | **0.9839** | **0.9795** | **0.9887** | **0.9840** |
| Gradient Boosting | 0.9007 | 0.8964 | 0.9069 | 0.9016 |
| Logistic Regression | 0.8705 | 0.8780 | 0.8614 | 0.8696 |

**Random Forest** achieved the best overall performance, with especially high Recall — critical in a medical context where missing an actual stroke case carries serious consequences.

## Key risk factors identified

- **Average glucose level** was the dominant driver of the first principal component (PCA loading ≈ 0.9997)
- **Age** and **BMI** were the next most influential factors (PCA2 loadings of 0.85 and 0.36)
- Stroke cases were concentrated in older individuals with higher glucose levels and higher BMI

## Tech stack

- Python — pandas, NumPy
- scikit-learn (KMeans, Agglomerative Clustering, DBSCAN, Gaussian Mixture, Logistic Regression, Random Forest, Gradient Boosting, PCA, StandardScaler)
- imbalanced-learn (SMOTE)
- TensorFlow / Keras
- Matplotlib, Seaborn (visualization)

## Repo contents

```
├── stroke_prediction.ipynb        # Full analysis notebook (EDA, preprocessing, clustering, classification)
├── Stroke_Prediction_Report.pdf   # Full written report
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/stroke-prediction.git
cd stroke-prediction
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), save it as `our_data.csv` in this folder, then open `stroke_prediction.ipynb` in Jupyter or Google Colab.

## What we'd improve next

- Improve the balance between precision and recall (current models favor recall)
- Explore additional ensemble/stacking methods used in related literature (e.g. LightGBM + SMOTE-ENN, which achieved AUC of 0.958 in published research)
- Add model interpretability tools (e.g. SHAP) to better explain individual predictions to clinicians

## Team

| Member | Focus area |
|---|---|
| Hanan Awadh | Literature review, unsupervised learning |
| Raghad Bawazeer | Literature review, supervised learning |
| Sharifa Mohammed | Literature review, unsupervised learning |
| Haneen Bamas | Literature review, unsupervised learning |
| Mais Mohammed | Literature review, supervised learning |
| Dhai Alshareef | Literature review, supervised learning |

## References

Full citation list available in the [report](Stroke_Prediction_Report.pdf).
