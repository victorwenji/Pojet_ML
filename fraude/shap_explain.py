# fraude/shap_explain.py
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

def explain(model_path, X_test):
    # Charger le modèle
    model = joblib.load(model_path)
    
    # Vérifier si le modèle a une liste de colonnes sauvegardée
    if hasattr(model, "feature_names_"):
        expected_cols = model.feature_names_
    else:
        # fallback : utiliser les colonnes de X_test
        expected_cols = X_test.columns.tolist()
    
    # Aligner X_test avec les colonnes attendues
    X_aligned = X_test.copy()
    for col in expected_cols:
        if col not in X_aligned.columns:
            # Ajouter une colonne manquante remplie de 0
            X_aligned[col] = 0
    # Supprimer les colonnes en trop
    X_aligned = X_aligned[expected_cols]
    
    # Créer l'explainer SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_aligned)
    
    # Vérifier si c'est un problème binaire
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
    
    # Graphiques SHAP
    plt.figure()
    shap.summary_plot(shap_vals, X_aligned, plot_type="bar", show=False)
    plt.savefig("static/plots/shap_bar.png")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_vals, X_aligned, show=False)
    plt.savefig("static/plots/shap_beeswarm.png")
    plt.close()

    return expected_cols  