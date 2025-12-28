import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from fraude.preprocessing import clean_data
from fraude.exploration import explore_class_distribution
from fraude.smote_model import train_smote
from fraude.threshold_tuning import threshold_tuning
from fraude.shap_explain import explain

def run_pipeline(csv_path):
    df = pd.read_csv(csv_path)
    df = clean_data(df)

    explore_class_distribution(df)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    model = train_smote(X_train, y_train)
    joblib.dump(model, "models/fraude_smote.pkl")

    y_prob = model.predict_proba(X_test)[:,1]
    threshold = threshold_tuning(y_test, y_prob)

    explain("models/fraude_smote.pkl", X_test)

    df["prediction"] = (model.predict_proba(X)[:,1] >= threshold).astype(int)
    return df.head(), threshold
