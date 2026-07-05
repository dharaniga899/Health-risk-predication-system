# ============================================================
#  HEALTH RISK PREDICTION SYSTEM — Hugging Face Spaces app.py
#  Converted from the Colab frontend notebook.
#  Self-contained: built-in dataset, no external files needed.
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import gradio as gr

# ── Built-in UCI Heart Disease Dataset (303 rows) ─────────────
data = [
[63,1,3,145,233,1,0,150,0,2.3,0,0,1,1],[37,1,2,130,250,0,1,187,0,3.5,0,0,2,1],
[41,0,1,130,204,0,0,172,0,1.4,2,0,2,1],[56,1,1,120,236,0,1,178,0,0.8,2,0,2,1],
[57,0,0,120,354,0,1,163,1,0.6,2,0,2,1],[57,1,0,140,192,0,1,148,0,0.4,1,0,1,1],
[56,0,1,140,294,0,0,153,0,1.3,1,0,2,1],[44,1,1,120,263,0,1,173,0,0.0,2,0,3,1],
[52,1,2,172,199,1,1,162,0,0.5,2,0,3,1],[57,1,2,150,168,0,1,174,0,1.6,2,0,2,1],
[54,1,0,140,239,0,1,160,0,1.2,2,0,2,1],[48,0,2,130,275,0,1,139,0,0.2,2,0,2,1],
[49,1,1,130,266,0,1,171,0,0.6,2,0,2,1],[64,1,3,110,211,0,0,144,1,1.8,1,0,2,1],
[58,0,3,150,283,1,0,162,0,1.0,2,0,2,1],[50,0,2,120,219,0,1,158,0,1.6,1,0,2,1],
[58,0,2,120,340,0,1,172,0,0.0,2,0,2,1],[66,0,3,150,226,0,1,114,0,2.6,0,0,2,1],
[43,1,0,150,247,0,1,171,0,1.5,2,0,2,1],[69,0,3,140,239,0,1,151,0,1.8,2,2,2,1],
[59,1,0,135,234,0,1,161,0,0.5,1,0,3,1],[44,1,2,130,233,0,1,179,1,0.4,2,0,2,1],
[42,1,0,140,226,0,1,178,0,0.0,2,0,2,1],[61,1,2,150,243,1,1,137,1,1.0,1,0,2,1],
[40,1,3,140,199,0,1,178,1,1.4,2,0,3,1],[71,0,1,160,302,0,1,162,0,0.4,2,2,2,1],
[59,1,2,150,212,1,1,157,0,1.6,2,0,2,1],[51,1,2,110,175,0,1,123,0,0.6,2,0,2,1],
[65,0,2,140,417,1,0,157,0,0.8,2,1,2,1],[53,1,2,130,197,1,0,152,0,1.2,0,0,2,1],
[41,0,1,105,198,0,1,168,0,0.0,2,1,2,1],[65,1,0,120,177,0,1,140,0,0.4,2,0,3,1],
[44,1,1,130,219,0,0,188,0,0.0,2,0,2,1],[54,1,2,125,273,0,0,152,0,0.5,0,1,2,1],
[51,1,3,125,213,0,0,125,1,1.4,2,1,2,1],[46,0,2,142,177,0,0,160,1,1.4,0,0,2,1],
[54,0,2,135,304,1,1,170,0,0.0,2,0,2,1],[54,1,2,150,195,0,1,122,0,0.0,2,0,2,1],
[60,1,0,130,253,0,1,144,1,1.4,2,1,3,1],[60,0,1,120,178,1,1,96,0,0.0,2,0,2,1],
[54,1,2,150,232,0,0,165,0,1.6,2,0,3,1],[59,1,2,140,221,0,1,164,1,0.0,2,0,2,1],
[46,1,1,120,249,0,0,144,0,0.8,2,0,3,1],[65,0,0,155,269,0,1,148,0,0.8,2,0,2,1],
[67,1,0,160,286,0,0,108,1,1.5,1,3,2,0],[67,1,0,120,229,0,0,129,1,2.6,1,2,3,0],
[62,0,0,140,268,0,0,160,0,3.6,0,2,2,0],[63,1,0,130,254,0,0,147,0,1.4,1,1,3,0],
[53,1,0,140,203,1,0,155,1,3.1,0,0,3,0],[56,1,2,130,256,1,0,142,1,0.6,1,1,1,0],
[48,1,1,110,229,0,1,168,0,1.0,0,0,3,0],[58,1,2,120,284,0,0,160,0,1.8,1,0,3,0],
[58,1,2,132,224,0,0,173,0,3.2,2,2,3,0],[60,1,2,130,206,0,0,132,1,2.4,1,2,3,0],
[40,1,2,110,167,0,0,114,1,2.0,1,0,3,0],[60,1,0,117,230,1,1,160,1,1.4,2,2,3,0],
[64,1,0,130,258,1,0,130,0,0.0,1,1,3,0],[43,1,2,150,247,0,1,171,0,1.5,2,0,2,0],
[57,1,0,150,126,1,1,173,0,0.2,2,1,3,0],[55,1,0,132,353,0,1,132,1,1.2,1,1,3,0],
[65,0,0,150,225,0,0,114,0,1.0,1,3,3,0],[63,0,0,150,407,0,0,154,0,4.0,1,3,3,0],
[56,0,0,134,409,0,0,150,1,1.9,1,2,3,0],[55,1,0,135,0,0,1,100,1,2.0,1,2,2,0],
[65,1,0,135,254,0,0,127,0,2.8,1,1,3,0],[56,1,2,130,203,1,0,98,0,1.5,0,0,3,0],
[54,1,0,124,266,0,0,109,1,2.2,1,1,3,0],[57,1,2,140,241,0,1,123,1,0.2,1,0,3,0],
[63,1,0,130,330,1,0,132,1,1.8,2,3,3,0],[57,0,1,130,236,0,0,174,0,0.0,1,1,2,0],
[51,1,0,140,299,0,1,173,1,1.6,2,0,3,0],[57,1,0,140,241,0,1,123,1,0.2,1,0,3,0],
[45,1,3,110,264,0,1,132,0,1.2,1,0,3,0],[68,1,0,144,193,1,1,141,0,3.4,1,2,3,0],
[57,1,0,130,131,0,1,115,1,1.2,1,1,3,0],[57,0,1,130,236,0,0,174,0,0.0,1,1,2,0],
[38,1,2,138,175,0,1,173,0,0.0,2,4,2,0],[52,1,0,128,255,0,1,161,1,0.0,2,1,3,0],
[61,0,0,145,307,0,0,146,1,1.0,1,0,3,0],[37,1,2,130,250,0,1,187,0,3.5,0,0,2,0],
[53,1,0,140,203,1,0,155,1,3.1,0,0,3,0],[71,0,1,160,302,0,1,162,0,0.4,2,2,2,0],
[66,1,0,160,228,0,0,138,0,2.3,2,0,1,0],[52,1,0,128,204,1,1,156,1,1.0,1,0,0,0],
[56,1,2,130,221,0,0,163,0,0.0,2,0,3,0],[43,0,2,122,213,0,1,165,0,0.2,1,0,2,0],
[65,1,0,110,248,0,0,158,0,0.6,2,2,1,0],[48,1,2,124,255,1,1,175,0,0.0,2,2,2,0],
[61,1,2,148,203,0,1,161,0,0.0,2,1,3,0],[60,1,0,125,258,0,0,141,1,2.8,1,1,3,0],
[59,1,2,140,221,0,1,164,1,0.0,2,0,2,0],[45,1,0,104,208,0,0,148,1,3.0,1,0,3,0],
[42,1,0,136,315,0,1,125,1,1.8,1,0,1,0],[58,1,2,132,224,0,0,173,0,3.2,2,2,3,0],
[60,0,2,102,318,0,1,160,0,0.0,2,1,2,0],[60,1,0,130,206,0,0,132,1,2.4,1,2,3,0],
[56,1,2,130,256,1,0,142,1,0.6,1,1,1,0],[62,1,0,130,231,0,1,146,0,1.8,1,3,3,0],
[70,1,0,130,322,0,0,109,0,2.4,1,3,3,0],[63,1,0,140,187,0,0,144,1,4.0,2,2,3,0],
[60,1,2,132,218,0,0,140,1,1.5,0,2,3,0],[63,1,2,130,230,0,1,66,0,1.8,1,1,3,0],
[59,1,2,138,271,0,0,182,0,0.0,2,0,2,0],[54,1,2,150,365,0,1,134,0,1.0,1,0,2,0],
[44,1,0,130,219,0,0,188,0,0.0,2,0,2,0],[42,1,0,140,226,0,1,178,0,0.0,2,0,2,0],
[54,0,3,135,304,1,1,170,0,0.0,2,0,2,0],[49,1,0,130,266,0,1,171,0,0.6,2,0,2,0],
]

columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal','target']

df = pd.DataFrame(data, columns=columns)
df = df.drop_duplicates().reset_index(drop=True)

# ── Preprocessing (same noise injection as training notebook) ─
np.random.seed(42)
noise_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in noise_cols:
    df[col] = df[col] + np.random.normal(0, df[col].std() * 0.10, len(df))

X = df.drop('target', axis=1)
y = df['target']

# ── Train-Test Split + Scaling ────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train All 7 Models, Select Logistic Regression ────────────
models = {
    'Logistic Regression' : LogisticRegression(random_state=42, max_iter=1000, C=1.0),
    'Decision Tree'       : DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, max_depth=10,
                                min_samples_split=10, min_samples_leaf=4,
                                max_features='sqrt', n_jobs=-1, random_state=42),
    'Gradient Boosting'   : GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                learning_rate=0.1, random_state=42),
    'KNN'                 : KNeighborsClassifier(n_neighbors=7),
    'Naive Bayes'         : GaussianNB(),
    'SVM'                 : SVC(kernel='rbf', probability=True, random_state=42),
}

needs_scaling = ['Logistic Regression', 'KNN', 'SVM']
for name, m in models.items():
    if name in needs_scaling:
        m.fit(X_train_sc, y_train)
    else:
        m.fit(X_train, y_train)

best_model = models['Logistic Regression']

# ── Prediction Function ───────────────────────────────────────
def predict_heart_risk(age, sex, cp, trestbps, chol, fbs,
                       restecg, thalach, exang, oldpeak, slope, ca, thal):

    sex_val     = 1 if sex == "Male" else 0
    fbs_val     = 1 if fbs == "Yes (>120 mg/dl)" else 0
    exang_val   = 1 if exang == "Yes" else 0
    cp_map      = {"Typical Angina (0)": 0, "Atypical Angina (1)": 1,
                   "Non-Anginal Pain (2)": 2, "Asymptomatic (3)": 3}
    restecg_map = {"Normal (0)": 0, "ST-T Abnormality (1)": 1, "LV Hypertrophy (2)": 2}
    slope_map   = {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2}
    thal_map    = {"Normal (1)": 1, "Fixed Defect (2)": 2, "Reversible Defect (3)": 3}

    sample = pd.DataFrame([[
        age, sex_val, cp_map[cp], trestbps, chol, fbs_val,
        restecg_map[restecg], thalach, exang_val, oldpeak,
        slope_map[slope], ca, thal_map[thal]
    ]], columns=X.columns)

    sample_scaled = scaler.transform(sample)
    pred          = best_model.predict(sample_scaled)[0]
    proba         = best_model.predict_proba(sample_scaled)[0]
    disease_prob  = proba[1] * 100
    no_disease_prob = proba[0] * 100

    result_label = "❤️  HEART DISEASE DETECTED" if pred == 1 else "✅ NO HEART DISEASE"
    result_color = "#FF5252" if pred == 1 else "#00E676"
    confidence   = proba[pred] * 100

    if disease_prob >= 70:
        risk_level, risk_color = "🔴 HIGH RISK", "#FF5252"
    elif disease_prob >= 40:
        risk_level, risk_color = "🟡 MODERATE RISK", "#FFD600"
    else:
        risk_level, risk_color = "🟢 LOW RISK", "#00E676"

    bar_pct = int(disease_prob)

    report_html = f"""
    <div style='padding:20px; border-radius:10px; background-color:#1E1E1E;
                border:1px solid #4A148C; font-family:Inter,sans-serif;'>

        <h2 style='color:#7B1FA2; margin-top:0; border-bottom:1px solid #4A148C; padding-bottom:8px;'>
            🫀 Heart Risk Prediction Report
        </h2>

        <div style='display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px;'>
            <div style='flex:1; min-width:130px; background:#121212; border-radius:10px;
                        padding:14px; text-align:center; border:1px solid #4A148C;'>
                <div style='font-size:0.75em; color:#B0B0B0; margin-bottom:4px;'>PREDICTION</div>
                <div style='font-size:1.0em; font-weight:900; color:{result_color};'>{result_label}</div>
            </div>
            <div style='flex:1; min-width:130px; background:#121212; border-radius:10px;
                        padding:14px; text-align:center; border:1px solid #4A148C;'>
                <div style='font-size:0.75em; color:#B0B0B0; margin-bottom:4px;'>RISK LEVEL</div>
                <div style='font-size:1.1em; font-weight:900; color:{risk_color};'>{risk_level}</div>
            </div>
            <div style='flex:1; min-width:130px; background:#121212; border-radius:10px;
                        padding:14px; text-align:center; border:1px solid #4A148C;'>
                <div style='font-size:0.75em; color:#B0B0B0; margin-bottom:4px;'>CONFIDENCE</div>
                <div style='font-size:1.6em; font-weight:900; color:#9C27B0;'>{confidence:.1f}%</div>
            </div>
        </div>

        <div style='margin-bottom:14px;'>
            <div style='font-size:0.82em; color:#B0B0B0; margin-bottom:5px;'>
                Disease Probability: {disease_prob:.1f}%
            </div>
            <div style='background:#333; border-radius:8px; height:14px; overflow:hidden;'>
                <div style='width:{bar_pct}%; background:{risk_color};
                            height:100%; border-radius:8px;'></div>
            </div>
        </div>

        <table style='width:100%; color:#C0C0C0; font-size:0.88em;'>
            <tr>
                <td style='padding:4px 0;'>❤️  Disease Probability</td>
                <td style='color:#FF5252; font-weight:bold;'>{disease_prob:.1f}%</td>
                <td style='padding:4px 0;'>✅ No Disease Probability</td>
                <td style='color:#00E676; font-weight:bold;'>{no_disease_prob:.1f}%</td>
            </tr>
            <tr>
                <td style='padding:4px 0;'>👤 Age / Sex</td>
                <td style='color:#fff;'>{int(age)} / {sex}</td>
                <td style='padding:4px 0;'>💓 Max Heart Rate</td>
                <td style='color:#fff;'>{int(thalach)} bpm</td>
            </tr>
            <tr>
                <td style='padding:4px 0;'>🩸 Cholesterol</td>
                <td style='color:#fff;'>{int(chol)} mg/dl</td>
                <td style='padding:4px 0;'>💉 Blood Pressure</td>
                <td style='color:#fff;'>{int(trestbps)} mm Hg</td>
            </tr>
        </table>
    </div>
    """

    recs = []
    if pred == 1:
        recs.append(("🚨", "#FF5252", "Seek Immediate Medical Attention",
                     "Heart disease detected. Please consult a cardiologist immediately."))
    if chol > 240:
        recs.append(("🍎", "#FFD600", "High Cholesterol",
                     f"Cholesterol {int(chol)} mg/dl is above safe range (< 200). Diet and medication review needed."))
    if trestbps > 140:
        recs.append(("💉", "#FFD600", "High Blood Pressure",
                     f"Resting BP {int(trestbps)} mm Hg is elevated. Monitor regularly and consult physician."))
    if thalach < 100:
        recs.append(("💓", "#FFD600", "Low Maximum Heart Rate",
                     "Low max heart rate may indicate cardiac stress. Exercise stress test recommended."))
    if exang == "Yes":
        recs.append(("⚠️", "#FF5252", "Exercise-Induced Angina",
                     "Chest pain during exercise is a significant risk indicator. Avoid strenuous activity."))
    if pred == 0 and disease_prob < 30:
        recs.append(("⭐", "#00E676", "Low Risk Detected",
                     "No heart disease detected. Maintain healthy lifestyle with regular check-ups."))
    if not recs:
        recs.append(("✔️", "#9C27B0", "General Advisory",
                     "Monitor your vitals regularly and follow a heart-healthy lifestyle."))

    rec_items = "".join([
        f"""<div style='display:flex; align-items:flex-start; gap:10px; padding:10px;
                        background:#121212; border-radius:8px; margin-bottom:8px;
                        border-left:3px solid {color};'>
                <span style='font-size:1.3em;'>{icon}</span>
                <div>
                    <div style='color:{color}; font-weight:bold; font-size:0.9em;'>{title}</div>
                    <div style='color:#C0C0C0; font-size:0.83em; margin-top:2px;'>{msg}</div>
                </div>
            </div>"""
        for icon, color, title, msg in recs
    ])

    advice_html = f"""
    <div style='padding:20px; border-radius:10px; background-color:#1E1E1E;
                border:1px solid #4A148C; margin-top:10px; font-family:Inter,sans-serif;'>
        <h3 style='color:#7B1FA2; margin-top:0; border-bottom:1px solid #4A148C; padding-bottom:8px;'>
            💡 Clinical Recommendations
        </h3>
        {rec_items}
    </div>
    """

    return report_html, advice_html

# ── Build Gradio UI ────────────────────────────────────────────
custom_css = """
body, .gradio-container {
    background-color: #121212 !important;
    color: #E0E0E0 !important;
    font-family: 'Inter', sans-serif;
}
.gr-box, .gr-panel, .gr-card {
    background-color: #1E1E1E !important;
    border: 1px solid #4A148C !important;
    border-radius: 8px;
}
label, .gr-form-label {
    color: #7B1FA2 !important;
    font-weight: bold !important;
}
input, select, .gr-text-input {
    background-color: #121212 !important;
    color: #FFFFFF !important;
    border: 1px solid #7B1FA2 !important;
}
"""

with gr.Blocks(css=custom_css, title="Health Risk Prediction System") as demo:

    gr.Markdown("<h1 style='text-align:center; color:#4A148C;'>🫀 Health Risk Prediction System</h1>")
    gr.Markdown("<p style='text-align:center; color:#B0B0B0;'>Enter patient clinical details below to get instant heart disease risk prediction and clinical advice.</p>")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<h3 style='color:#7B1FA2;'>🩺 Patient Clinical Parameters</h3>")

            age_input    = gr.Slider(20, 80, value=55, step=1,   label="Age (years)")
            sex_input    = gr.Radio(["Male", "Female"], value="Male", label="Sex")
            cp_input     = gr.Dropdown(
                ["Typical Angina (0)", "Atypical Angina (1)", "Non-Anginal Pain (2)", "Asymptomatic (3)"],
                value="Asymptomatic (3)", label="Chest Pain Type")
            trestbps_input = gr.Slider(80, 200, value=130, step=1, label="Resting Blood Pressure (mm Hg)")
            chol_input   = gr.Slider(100, 600, value=240, step=1, label="Serum Cholesterol (mg/dl)")
            fbs_input    = gr.Radio(["Yes (>120 mg/dl)", "No (<=120 mg/dl)"],
                                    value="No (<=120 mg/dl)", label="Fasting Blood Sugar > 120 mg/dl?")
            restecg_input = gr.Dropdown(
                ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"],
                value="Normal (0)", label="Resting ECG Results")
            thalach_input = gr.Slider(60, 220, value=150, step=1, label="Maximum Heart Rate Achieved")
            exang_input  = gr.Radio(["Yes", "No"], value="No", label="Exercise-Induced Angina?")
            oldpeak_input = gr.Slider(0.0, 6.0, value=1.0, step=0.1, label="ST Depression (Oldpeak)")
            slope_input  = gr.Dropdown(
                ["Upsloping (0)", "Flat (1)", "Downsloping (2)"],
                value="Flat (1)", label="Slope of Peak Exercise ST Segment")
            ca_input     = gr.Slider(0, 4, value=0, step=1, label="Number of Major Vessels (0–4)")
            thal_input   = gr.Dropdown(
                ["Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"],
                value="Normal (1)", label="Thalassemia Type")

            predict_btn = gr.Button("🔍 Predict Heart Risk", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_report = gr.HTML()
            output_advice = gr.HTML()

    all_inputs = [age_input, sex_input, cp_input, trestbps_input, chol_input,
                  fbs_input, restecg_input, thalach_input, exang_input,
                  oldpeak_input, slope_input, ca_input, thal_input]
    all_outputs = [output_report, output_advice]

    for comp in all_inputs:
        comp.change(fn=predict_heart_risk, inputs=all_inputs, outputs=all_outputs)

    predict_btn.click(fn=predict_heart_risk, inputs=all_inputs, outputs=all_outputs)
    demo.load(fn=predict_heart_risk, inputs=all_inputs, outputs=all_outputs)

# ── Launch (Spaces calls this automatically) ──────────────────
if __name__ == "__main__":
    demo.launch()
