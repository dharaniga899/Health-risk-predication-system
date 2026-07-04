# ❤️ Heart Disease Risk Prediction System

A machine learning system that predicts a patient's risk of heart disease from clinical parameters. Built, trained, and evaluated entirely in **Google Colab** using the classic UCI-style Heart Disease dataset.

---

## 📌 Overview

This project compares **seven classification algorithms** on a heart disease dataset and selects **Logistic Regression** as the final production model based on its balance of accuracy, recall, and stability across cross-validation folds. The trained model, scaler, and feature schema are exported so predictions can be made on new patient data.

---

## 🧠 Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.12 |
| Environment | Google Colab |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Persistence | Joblib, JSON |

---

## 📂 Dataset — `heart.csv`

| Property | Value |
|---|---|
| Raw shape | 1025 rows × 14 columns |
| After removing duplicates | 302 rows × 14 columns (723 duplicate rows dropped) |
| Target balance (post-cleaning) | 164 Disease (54.3%) / 138 No Disease (45.7%) — nearly balanced |
| Missing values | None |

### Features

| Column | Description |
|---|---|
| `age` | Patient age in years |
| `sex` | 1 = Male, 0 = Female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = Yes, 0 = No) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `ca` | Number of major vessels colored by fluoroscopy (0–4) |
| `thal` | Thalassemia (1 = Normal, 2 = Fixed defect, 3 = Reversible defect) |
| `target` | 1 = Heart Disease, 0 = No Heart Disease |

---

## ⚙️ Project Workflow

1. **Environment setup** — verify Python/library versions, mount Google Drive.
2. **Data loading** — read `heart.csv` from Google Drive into a Pandas DataFrame.
3. **Data inspection** — check shape, dtypes, nulls, duplicates, and summary statistics.
4. **Data cleaning** — drop 723 duplicate rows (1025 → 302 rows).
5. **Exploratory Data Analysis (EDA)**
   - Target distribution (bar chart + pie chart)
   - Feature histograms
   - Categorical features vs. target (grouped bar charts)
   - Correlation heatmap
   - Boxplots of continuous features by target class
6. **Preprocessing**
   - Adds small Gaussian noise (10% of std) to continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) to simulate real-world measurement variability and reduce overfitting to clean clinical boundaries.
   - Train/test split: 80/20, stratified on target (241 train / 61 test rows).
   - Feature scaling with `StandardScaler` (fit on train, applied to test).
7. **Model training & comparison** — trains and evaluates 7 classifiers:
   - Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, Naive Bayes, SVM
8. **Model selection** — Logistic Regression is chosen as the final model.
9. **Evaluation** — classification report, confusion matrix, ROC curve, 5-fold cross-validation.
10. **Model interpretability** — logistic regression coefficients plotted to show which features increase/decrease risk.
11. **Model export** — model, scaler, and feature list saved for reuse.
12. **Inference function** — `predict_heart_risk()` loads the saved artifacts and predicts on new patient data.

---

## 📊 Model Comparison Results

Evaluated on the 61-row held-out test set:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Naive Bayes | 0.7869 | 0.8333 | 0.7576 | 0.7937 | **0.8842** |
| **Logistic Regression (selected)** | 0.7869 | 0.7778 | **0.8485** | **0.8116** | 0.8626 |
| KNN | **0.8033** | 0.8000 | 0.8485 | **0.8235** | 0.8582 |
| Random Forest | 0.7869 | 0.7941 | 0.8182 | 0.8060 | 0.8528 |
| SVM | 0.7541 | 0.7500 | 0.8182 | 0.7826 | 0.8409 |
| Gradient Boosting | 0.7541 | 0.7812 | 0.7576 | 0.7692 | 0.8279 |
| Decision Tree | 0.7541 | 0.8000 | 0.7273 | 0.7619 | 0.7933 |

> **Why Logistic Regression was chosen:** although KNN had marginally higher accuracy and Naive Bayes had a higher ROC-AUC, Logistic Regression gave the best recall (0.8485) — i.e., it misses the fewest actual heart disease cases, which matters most in a medical screening context — while remaining simple, interpretable, and well-calibrated.

### Final Logistic Regression Metrics

| Metric | Score |
|---|---|
| Test Accuracy | 78.69% |
| F1 Score | 0.8116 |
| ROC-AUC | 0.8626 |

**Confusion Matrix (test set, n = 61):**

| | Predicted No Disease | Predicted Disease |
|---|---|---|
| **Actual No Disease** | 20 (TN) | 8 (FP) |
| **Actual Disease** | 5 (FN) | 28 (TP) |

**5-Fold Cross-Validation (ROC-AUC):** Mean = **0.8943**, Std = 0.0497 — indicates a stable model with low variance across folds.

### Feature Impact (Logistic Regression Coefficients)

Top features increasing risk: `cp` (chest pain type), `thalach` (max heart rate), `slope`, `restecg`, `age`.
Top features decreasing risk: `sex`, `ca`, `thal`, `exang`, `trestbps`, `oldpeak`, `chol`, `fbs`.

---

## 🚀 How to Run

1. Open `Health_risk_predication_system3008Untitled0__1_.ipynb` in Google Colab.
2. Upload `heart.csv` to your Google Drive (default expected path in the notebook):
   ```
   /content/drive/MyDrive/datasetheart/heart.csv
   ```
   Or edit the `path` variable in the data-loading cell to match your own location.
3. Run all cells in order: **Runtime → Run all**. The notebook will prompt you to authorize Google Drive access.
4. The notebook will output EDA plots, model comparison metrics, and finally save the trained model artifacts to your Drive:
   - `heart_lr_model.pkl`
   - `heart_lr_scaler.pkl`
   - `heart_lr_feature_columns.json`

---

## 🔮 Making Predictions on New Data

The notebook defines a ready-to-use `predict_heart_risk()` function:

```python
predict_heart_risk({
    'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145,
    'chol': 233, 'fbs': 1, 'restecg': 0,
    'thalach': 150, 'exang': 0, 'oldpeak': 2.3,
    'slope': 0, 'ca': 0, 'thal': 1
})
```

It loads the saved model, scaler, and feature schema, then returns the prediction, confidence, and class probabilities, e.g.:

```
Prediction  : ❤️  HEART DISEASE DETECTED
Confidence  : 83.59%
Disease Prob: 83.59%  |  No Disease Prob: 16.41%
```

---

## 📁 Project Structure

```
heart-disease-risk-prediction/
│
├── Health_risk_predication_system3008Untitled0__1_.ipynb   # Main Colab notebook
├── heart.csv                                                # Dataset
├── heart_lr_model.pkl                                       # Saved Logistic Regression model (generated)
├── heart_lr_scaler.pkl                                      # Saved StandardScaler (generated)
├── heart_lr_feature_columns.json                            # Feature column order (generated)
└── README.md                                                # Project documentation
```

---

## 🔮 Future Improvements

- Note: the current train/test split is small (241/61 rows) after duplicate removal — a larger or additional dataset would improve reliability.
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for Logistic Regression and other top performers (KNN, Naive Bayes).
- Deploy the model as an interactive web app (Streamlit/Flask) using the saved `.pkl` artifacts.
- Add SHAP explainability on top of the existing coefficient analysis.
- Investigate why KNN/Naive Bayes slightly outperform on certain metrics — consider an ensemble/voting classifier.

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**. It is not a certified medical diagnostic tool and should **not** be used as a substitute for professional medical advice. Always consult a qualified healthcare provider for medical concerns.

---

## 👤 Author
DHARANIGA V.R
---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — feel free to use and modify it with attribution.
