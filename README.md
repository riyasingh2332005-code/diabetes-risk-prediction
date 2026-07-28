# Diabetes Risk Prediction System

A machine learning project that predicts diabetes risk from health survey indicators, deployed as an interactive Streamlit web app.

**Live demo:** _add your Streamlit Cloud URL here after deploying_

## Overview

This project trains a Logistic Regression model on CDC BRFSS health survey data to estimate diabetes risk, then wraps it in a user-friendly web app with visual results, personalized health tips, and a downloadable PDF report.

## Dataset

- **Source:** [Diabetes Health Indicators (BRFSS 2015-2023)](https://www.kaggle.com/datasets/vanthieu/diabetes-health-indicators-brfss-2015-2023) — Kaggle, CC0 Public Domain
- **File used:** `cdc_brfss_diabetes_2023.csv`
- **Size:** 238,964 rows after removing duplicates, 17 features + 1 target
- **Target:** `Diabetes_binary` (0 = no diabetes, 1 = diagnosed diabetes)
- **Class balance:** ~84% no diabetes / ~16% diabetes — an imbalanced dataset, which shaped the modeling approach below

### Features used

HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income

## Problem Statement

Type 2 diabetes is often undiagnosed until complications appear. A model that flags at-risk individuals from simple, self-reported health indicators can support earlier screening — even an imperfect model has real value as a first-pass triage step before clinical testing.

## Methodology

1. **Data cleaning:** removed exact duplicate rows; verified no missing values.
2. **EDA:** explored relationships between individual features and diabetes rate (see Key Findings below).
3. **Modeling:** compared Logistic Regression, Random Forest, and XGBoost, using SMOTE and class weighting to address class imbalance.
4. **Leakage-free threshold tuning:** the training data was split further into a train/validation set. The decision threshold was tuned on the validation set only, then evaluated once on a held-out test set — avoiding the common mistake of tuning and evaluating on the same data.
5. **Interpretation:** examined scaled feature coefficients to identify the strongest predictors, and used them to power in-app risk factor explanations.

## Key EDA Findings

- **BMI:** Diabetic respondents show a visibly higher BMI distribution than non-diabetic respondents.
- **Age:** Diabetes rate rises sharply with age, from ~2% in the youngest bracket (18-24) to a peak of ~24-25% around ages 70-79, before a slight dip in the 80+ group.
- **General Health (GenHlth):** The strongest single relationship in the data — diabetes rate climbs from ~4% among those reporting "excellent" health to ~38% among those reporting "poor" health, a nearly 10x difference.
- **Class imbalance:** ~84% of respondents report no diabetes, meaning a naive model predicting "no diabetes" for everyone would score ~84% accuracy — which is why this project prioritizes recall and F1-score over raw accuracy.

## Model Comparison

| Model | Precision (Diabetic) | Recall (Diabetic) | F1-score (Diabetic) |
|---|---|---|---|
| Logistic Regression (class-weighted, threshold=0.5) | 0.32 | 0.74 | 0.45 |
| Random Forest (class-weighted only) | 0.39 | 0.14 | 0.21 |
| Random Forest (SMOTE, tuned depth) | 0.39 | 0.47 | 0.43 |
| XGBoost (SMOTE) | 0.54 | 0.18 | 0.27 |
| **Logistic Regression (threshold=0.4) — final choice** | 0.28 | **0.84** | 0.42 |

A consistent pattern held across all three algorithm families: tree-based models (Random Forest, XGBoost) leaned toward higher precision but much lower recall, while Logistic Regression favored recall. Since missing an actual diabetic case (a false negative) is more costly than a false alarm in a screening context, **Logistic Regression with a 0.4 decision threshold** was chosen as the final model — deliberately prioritizing recall over the model with the single highest F1-score.

**Final test set results** (threshold tuned on a separate validation split, then evaluated once on test data):

```
              precision    recall  f1-score   support
         0.0       0.95      0.59      0.72     39975
         1.0       0.28      0.84      0.42      7818
```

Out of 7,818 actual diabetic cases in the test set, the model correctly identified ~6,575 (84%), missing ~1,243.

## Feature Importance

After standardizing features for coefficient comparison (an initial unscaled comparison had incorrectly ranked `CholCheck` as the top predictor — a scaling artifact, since raw coefficients aren't comparable across features on different numeric scales), the strongest positive predictors of diabetes were:

1. Age
2. General Health rating (GenHlth)
3. BMI
4. High Blood Pressure
5. High Cholesterol

**Income** showed the strongest negative association — higher income was linked to lower diabetes likelihood, likely reflecting better healthcare access and lifestyle factors.

## The App

Built with Streamlit, the app includes:

- **Two assessment modes:** a Quick Assessment (5 key questions) and a Full Assessment (all 17 features), so users can trade off speed for precision.
- **BMI calculator:** enter BMI directly, or calculate it from height and weight, with an automatic WHO category label (Underweight/Normal/Overweight/Obese).
- **Human-readable age selection:** an age-range dropdown (e.g. "50-54") instead of the raw coded BRFSS age bracket.
- **Live preview:** an instant, rough risk estimate that updates as inputs change, before running the full analysis.
- **Visual results:** a color-coded gauge chart, a radar chart of the user's health profile, and a bar chart comparing their risk to their age group's average (from the EDA findings above).
- **Risk factor breakdown:** the top contributing factors behind each individual prediction, derived from the model's coefficients.
- **Personalized health tips:** rule-based suggestions generated from the user's specific inputs (e.g. BMI, activity level, smoking status).
- **Emergency guidance:** a clear "consult a healthcare professional" message when estimated risk is high.
- **Downloadable PDF report** summarizing the result, key inputs, and top risk factors.

## Project Structure

```
diabetes-app/
├── cdc_brfss_diabetes_2023.csv     # raw dataset
├── cleaned_data.csv                # cleaned dataset (generated by 01_explore.py)
├── 01_explore.py                   # data loading, cleaning, target balance check
├── 02_eda.py                       # exploratory plots (BMI, Age, GenHlth vs diabetes)
├── 03_train_model.py               # train/val/test split, model training, threshold tuning, saves model
├── diabetes_model.pkl              # trained Logistic Regression model
├── app.py                          # Streamlit web app
├── requirements.txt
└── README.md
```

## How to Run Locally

1. Clone this repository and open the folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Rebuild the model from scratch:
   ```bash
   python 01_explore.py
   python 02_eda.py
   python 03_train_model.py
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (include `app.py`, `diabetes_model.pkl`, and `requirements.txt`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository and branch, and set the main file to `app.py`.
4. Click **Deploy** — you'll get a public URL anyone can visit.

## What I'd Improve With More Time

- Cross-validation (5-fold) to confirm results are stable across different data splits.
- Combine all five BRFSS survey years (2015-2023) instead of using only 2023, for a larger and more temporally robust training set.
- Try `scale_pos_weight` tuning in XGBoost specifically targeting recall, as an alternative to SMOTE.
- Replace the rule-based health tips with an LLM-generated personalized summary.
- Add a symptom/history tracker so returning users can see their risk trend over time.

## Tools Used

Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, imbalanced-learn (SMOTE), XGBoost, Streamlit, Plotly, fpdf2

## Disclaimer

This tool provides an estimate based on a machine learning model trained on survey data. It is for educational purposes only and is **not** a medical diagnosis. Please consult a healthcare professional for accurate testing and advice.