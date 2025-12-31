from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Liste cruciale pour éviter que le SMOTE ne "triche"
EXCLUDE_COLS = [
    "score_risque_marchand",
    "nb_tentatives_echouees",
    "montant_total_24h",
    "nb_trans_24h"
]

def train_smote(X_train, y_train):
    # 1. On ne garde que les colonnes saines pour l'entraînement
    cols_to_drop = [c for c in EXCLUDE_COLS if c in X_train.columns]
    X_train_safe = X_train.drop(columns=cols_to_drop)
    
    # 2. SMOTE génère des voisins proches uniquement sur les variables comportementales
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_safe, y_train)

    # 3. Entraînement
    model = RandomForestClassifier(random_state=42)
    model.fit(X_res, y_res)
    
    model.feature_names_in_ = X_train_safe.columns.tolist()
    return model

def evaluate_model(model, X_test, y_test):
    # Aligner le test sur les mêmes colonnes que le train (sans le leakage)
    X_test_safe = X_test[model.feature_names_in_]
    
    y_pred = model.predict(X_test_safe)
    y_prob = model.predict_proba(X_test_safe)[:,1]
    
    return {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }