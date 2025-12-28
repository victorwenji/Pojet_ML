import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Colonnes à exclure pour éviter fuite de données
EXCLUDE_COLS = [
    "is_fraud",
    "score_risque_marchand",
    "nb_tentatives_echouees",
    "montant_total_24h",
    "nb_trans_24h"
]

def train_baseline_model(X_train, y_train):
    # Retirer colonnes à exclure
    X_train_safe = X_train.drop(columns=[c for c in EXCLUDE_COLS if c in X_train.columns])
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_safe, y_train)
    # Sauvegarder les colonnes utilisées
    model.feature_names_ = X_train_safe.columns.tolist()
    return model

def evaluate_model(model, X_test, y_test):
    # Aligner colonnes test avec celles du train
    X_test_safe = X_test.copy()
    for col in model.feature_names_in_:
        if col not in X_test_safe.columns:
            X_test_safe[col] = 0
    X_test_safe = X_test_safe[model.feature_names_in_]


    y_pred = model.predict(X_test_safe)
    y_prob = model.predict_proba(X_test_safe)[:,1]
    metrics = {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    return metrics

def save_model(model, path):
    joblib.dump(model, path)

