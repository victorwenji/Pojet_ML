import os
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from fraude.preprocessing import clean_data
from fraude.baseline import train_baseline_model, evaluate_model
from fraude.class_weight import train_rf_class_weight, train_gb_class_weight
from fraude.smote_model import train_smote, evaluate_model as eval_smote
from fraude.threshold_tuning import threshold_tuning

# Chemins
DATA_PATH = "data/raw/transactions.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Charger et nettoyer
df = pd.read_csv(DATA_PATH)
df_clean = clean_data(df)

# 2. Split train/test
target_col = "is_fraud"
X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entraîner Baseline
baseline_model = train_baseline_model(X_train, y_train)
joblib.dump(baseline_model, os.path.join(MODELS_DIR, "baseline.pkl"))
baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
print("Baseline:", baseline_metrics)

# 4. Random Forest class_weight
rf_model = train_rf_class_weight(X_train, y_train)
joblib.dump(rf_model, os.path.join(MODELS_DIR, "rf_class_weight.pkl"))
rf_metrics = evaluate_model(rf_model, X_test, y_test)
print("RF class_weight:", rf_metrics)

# 5. Gradient Boosting class_weight
gb_model = train_gb_class_weight(X_train, y_train)
joblib.dump(gb_model, os.path.join(MODELS_DIR, "gb_class_weight.pkl"))
gb_metrics = evaluate_model(gb_model, X_test, y_test)
print("GB class_weight:", gb_metrics)

# 6. SMOTE + Random Forest
smote_model = train_smote(X_train, y_train)
joblib.dump(smote_model, os.path.join(MODELS_DIR, "smote.pkl"))
smote_metrics = eval_smote(smote_model, X_test, y_test)
print("SMOTE:", smote_metrics)

# 7. Calcul et sauvegarde du meilleur seuil (threshold tuning)
y_prob = smote_model.predict_proba(X_test)[:, 1]
best_thresh, precisions, recalls, f1_scores, thresholds = threshold_tuning(y_test, y_prob)

# Sauvegarder le meilleur seuil
with open(os.path.join(MODELS_DIR, "best_threshold.json"), "w") as f:
    json.dump({"best_threshold": float(best_thresh)}, f)
print(f"Best threshold sauvegardé: {best_thresh}")

print("Tous les modèles ont été entraînés et sauvegardés dans le dossier 'models/'")
