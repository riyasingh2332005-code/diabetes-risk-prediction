import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Cleaned data load 
df = pd.read_csv('cleaned_data.csv')

# X aur y define 
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

print("Train:", X_train_final.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# Model training
baseline_final = LogisticRegression(class_weight='balanced', max_iter=1000)
baseline_final.fit(X_train_final, y_train_final)

# Threshold tuning 
val_probs = baseline_final.predict_proba(X_val)[:, 1]

print("\nThreshold tuning results (on validation set):")
for t in [0.3, 0.35, 0.4, 0.45, 0.5]:
    val_pred = (val_probs >= t).astype(int)
    report = classification_report(y_val, val_pred, output_dict=True)
    print(f"Threshold {t}: Precision={report['1.0']['precision']:.2f}, "
          f"Recall={report['1.0']['recall']:.2f}, F1={report['1.0']['f1-score']:.2f}")

# Final threshold 
final_threshold = 0.4

test_probs = baseline_final.predict_proba(X_test)[:, 1]
y_pred_final = (test_probs >= final_threshold).astype(int)

print(f"\nFinal test results (threshold={final_threshold}):")
print(classification_report(y_test, y_pred_final))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_final)
print("\nConfusion Matrix:\n", cm)

# Model saving
joblib.dump(baseline_final, 'diabetes_model.pkl')
print("\nModel saved as diabetes_model.pkl")