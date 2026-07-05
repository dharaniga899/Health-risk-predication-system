**Heart Disease Risk Prediction System**

A comprehensive **Machine Learning-based Heart Disease Risk Prediction System** that analyzes a patient's clinical parameters to estimate the likelihood of heart disease. The project compares multiple classification algorithms and selects **Logistic Regression** as the final model based on its performance, interpretability, and reliability. It features an interactive **Gradio-based web application** for real-time risk prediction, confidence scoring, and personalized clinical recommendations. The entire project—including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment—was developed in **Google Colab** using the Heart Disease Dataset**.

## 🌐 Live Demo

Try the app live on Hugging Face Spaces: **[Heart Disease Risk Prediction](https://huggingface.co/spaces/Dharaniga3008/Heart-disease-risk-prediction)**

---

## 📌 Overview

This project has two parts:

1. **Model training & comparison notebook** — compares **seven classification algorithms** and selects **Logistic Regression** as the final production model based on its balance of accuracy, recall, and stability across cross-validation folds.
2. **Frontend notebook (Gradio app)** — a dark-themed, interactive web interface where a user enters a patient's clinical details (via sliders, dropdowns, and radio buttons) and instantly gets a heart disease risk prediction along with clinical recommendations.


---

## 🧠 Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.12 |
| Environment | Google Colab |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Frontend / UI | Gradio |
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

## 🖥️ Frontend — Interactive Gradio App

The `frontend_health_risk_prediction.ipynb` notebook wraps the trained Logistic Regression model in a **dark-themed Gradio web interface** for real-time predictions — no coding needed to use it once launched.
 



**Inputs (Patient Clinical Parameters):**
- Age, Sex, Chest Pain Type, Resting Blood Pressure, Serum Cholesterol
- Fasting Blood Sugar, Resting ECG, Maximum Heart Rate, Exercise-Induced Angina
- ST Depression (Oldpeak), Slope, Number of Major Vessels, Thalassemia Type

**Outputs:**
- 🔍 **Prediction Report** — risk result, confidence %, disease/no-disease probability, and key vitals summary
- 💡 **Clinical Recommendations** — dynamic advice based on the input values, e.g.:
  - High cholesterol / blood pressure warnings
  - Low max heart rate flag
  - Exercise-induced angina alert
  - General advisory for low-risk patients


### Running the Gradio App

1. Open `frontend_health_risk_prediction.ipynb` in Google Colab.
2. Run the single cell — it installs Gradio, trains all 7 models on the built-in dataset (no Drive mount needed), and launches the app inline:
   ```python
   demo.launch(debug=False, inline=True, share=False)
   ```
3. To get a public shareable link instead of an inline Colab preview, change to `share=True`.

---

## 🚀 How to Run — Training Notebook

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
Health-risk-predication-system/
│
├── notebooks/
│   ├── Health_risk_predication_system3008Untitled0__1_.ipynb   # Model training & comparison
│   └── frontend_health_risk_prediction.ipynb                    # Gradio UI app
│
├── data/
│   └── heart.csv                                                 # Datasets
│
├── requirements.txt                                              # Python dependencies
├── LICENSE                                                       # MIT License
└── README.md                                                     # Project documentation
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

**[DHARANIGA V.R]**
- LinkedIn:[www.linkedin.com/in/dharaniga-v-r-abb485387]
- Email: [dharanigaraj@gmail.com]
---
## 📄 License

This project is licensed under the [MIT License](LICENSE) — feel free to use and modify it with attribution.

## 📸 Application Preview

<img width="1300" height="1038" alt="app_screenshot_red" src="https://github.com/user-attachments/assets/d8615459-60b5-4a4b-8f7f-d6fca9097242" />
