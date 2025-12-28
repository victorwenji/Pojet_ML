import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def train_rf_class_weight(X_train, y_train):
    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gb_class_weight(X_train, y_train):
    model = GradientBoostingClassifier(random_state=42)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    return {
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

def save_model(model, path):
    joblib.dump(model, path)
