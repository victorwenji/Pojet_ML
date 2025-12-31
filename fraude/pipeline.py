import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from fraude.preprocessing import clean_data
from fraude.exploration import explore_class_distribution
from fraude.smote_model import train_smote
from fraude.threshold_tuning import threshold_tuning
from fraude.shap_explain import explain

def run_pipeline(csv_path):
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(csv_path)
    df_clean = clean_data(df)

    explore_class_distribution(df_clean)

    X = df_clean.drop(columns=["is_fraud"])
    y = df_clean["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = train_smote(X_train, y_train)
    joblib.dump(model, "models/fraude_smote.pkl")

    y_prob_test = model.predict_proba(X_test)[:,1]
    threshold = threshold_tuning(y_test, y_prob_test)
    
    with open("models/threshold.json", "w") as f:
        json.dump({"best_threshold": float(threshold)}, f)

    explain("models/fraude_smote.pkl", X_test)

    X_test_result = X_test.copy()
    X_test_result["true_label"] = y_test
    X_test_result["prediction"] = (y_prob_test >= threshold).astype(int)
    
    print(f"Pipeline terminée. Seuil optimal : {threshold}")
    return X_test_result.head(), threshold