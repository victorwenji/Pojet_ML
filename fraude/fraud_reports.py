import pandas as pd
import numpy as np
import joblib
import shap
import os

def generate_fraud_reports(model_path, X_test, y_test, y_prob, threshold, top_n=10):
    """
    Génère un CSV avec les cas de fraude les plus suspects avec scores de risque
    """
    model = joblib.load(model_path)
    
    # Prédictions
    y_pred = (y_prob >= threshold).astype(int)
    
    # Créer DataFrame résultats
    df_results = X_test.copy()
    df_results['vraie_fraude'] = y_test.values
    df_results['prediction'] = y_pred
    df_results['probabilite_fraude'] = y_prob
    df_results['score_risque'] = (y_prob * 100).round(2)
    
    # Filtrer les fraudes détectées
    fraud_cases = df_results[df_results['prediction'] == 1].copy()
    
    if len(fraud_cases) == 0:
        print("⚠️ Aucune fraude détectée avec le seuil actuel")
        # Créer un rapport vide
        empty_report = pd.DataFrame({
            'score_risque': [],
            'probabilite_fraude': [],
            'statut': [],
            'top_features_risque': []
        })
        output_path = "static/reports/fraud_reports.csv"
        os.makedirs("static/reports", exist_ok=True)
        empty_report.to_csv(output_path)
        return empty_report
    
    # Trier par probabilité décroissante
    fraud_cases = fraud_cases.sort_values('probabilite_fraude', ascending=False)
    
    # Top N cas
    top_frauds = fraud_cases.head(top_n).copy()
    
    print(f"📋 Analyse de {len(top_frauds)} cas de fraude...")
    
    # Analyse SHAP pour chaque cas
    try:
        explainer = shap.TreeExplainer(model)
        
        # Aligner colonnes avec le modèle
        X_aligned = align_features(
            top_frauds.drop(columns=['vraie_fraude', 'prediction', 
                                    'probabilite_fraude', 'score_risque'], errors='ignore'),
            model
        )
        
        shap_values = explainer.shap_values(X_aligned)
        
        # Gérer différents formats de shap_values
        if isinstance(shap_values, list):
            # Classification binaire - prendre classe 1 (fraude)
            shap_values = shap_values[1]
        
        # Vérifier la forme de shap_values
        if len(shap_values.shape) == 1:
            # Si 1D, transformer en 2D
            shap_values = shap_values.reshape(1, -1)
        
        # Extraire top 3 features par transaction
        top_features_list = []
        for i in range(len(X_aligned)):
            try:
                # CORRECTION: Utiliser .ravel() pour forcer 1D
                if len(shap_values.shape) > 1:
                    shap_row = shap_values[i].ravel()
                else:
                    shap_row = shap_values
                
                # S'assurer que shap_row et colonnes ont la même longueur
                if len(shap_row) != len(X_aligned.columns):
                    print(f"⚠️ Longueur mismatch: shap={len(shap_row)}, features={len(X_aligned.columns)}")
                    top_features_list.append("Analyse non disponible")
                    continue
                
                feature_importance = pd.DataFrame({
                    'feature': X_aligned.columns.tolist(),
                    'shap_value': shap_row.tolist()
                }).sort_values('shap_value', ascending=False)
                
                top_3 = feature_importance.head(3)['feature'].tolist()
                top_features_list.append(', '.join(top_3))
            except Exception as e:
                print(f"⚠️ Erreur analyse transaction {i}: {e}")
                top_features_list.append("Analyse non disponible")
        
        top_frauds.loc[:, 'top_features_risque'] = top_features_list
        
    except Exception as e:
        print(f"⚠️ Erreur SHAP: {e}")
        print("Génération du rapport sans analyse SHAP détaillée...")
        top_frauds.loc[:, 'top_features_risque'] = "Analyse SHAP non disponible"
    
    # Statut de validation
    top_frauds.loc[:, 'statut'] = top_frauds['vraie_fraude'].apply(
        lambda x: '✅ Vraie Fraude' if x == 1 else '❌ Fausse Alerte'
    )
    
    # Sélectionner colonnes importantes pour le rapport
    report_cols = ['score_risque', 'probabilite_fraude', 'statut', 'top_features_risque']
    
    # Ajouter colonnes métier si disponibles
    business_cols = ['montant_transaction', 'type_transaction', 'categorie_marchand', 
                     'pays_transaction', 'mode_paiement']
    for col in business_cols:
        if col in top_frauds.columns:
            report_cols.insert(0, col)
    
    # Filtrer les colonnes existantes
    report_cols = [col for col in report_cols if col in top_frauds.columns]
    
    final_report = top_frauds[report_cols].reset_index(drop=True)
    final_report.index += 1
    final_report.index.name = 'ID_Transaction'
    
    # Sauvegarder
    os.makedirs("static/reports", exist_ok=True)
    output_path = "static/reports/fraud_reports.csv"
    final_report.to_csv(output_path)
    
    print(f"✅ Rapport de fraude généré: {output_path}")
    print(f"   - {len(final_report)} transactions analysées")
    print(f"   - Vraies fraudes: {(top_frauds['vraie_fraude'] == 1).sum()}")
    print(f"   - Fausses alertes: {(top_frauds['vraie_fraude'] == 0).sum()}")
    
    return final_report


def align_features(X, model):
    """
    Aligne les features avec celles du modèle
    Gère les cas où le modèle a été entraîné avec des colonnes différentes
    """
    # Essayer différentes façons de récupérer les noms de features
    expected_cols = None
    
    if hasattr(model, 'feature_names_'):
        expected_cols = model.feature_names_
    elif hasattr(model, 'feature_names_in_'):
        expected_cols = model.feature_names_in_
    elif hasattr(model, 'feature_name_'):
        expected_cols = model.feature_name_
    else:
        # Si aucune information, retourner X tel quel
        print("⚠️ Impossible de détecter les features du modèle, utilisation des colonnes actuelles")
        return X
    
    X_aligned = X.copy()
    
    # Ajouter les colonnes manquantes
    for col in expected_cols:
        if col not in X_aligned.columns:
            X_aligned[col] = 0
    
    # Supprimer les colonnes en trop
    extra_cols = [col for col in X_aligned.columns if col not in expected_cols]
    if extra_cols:
        X_aligned = X_aligned.drop(columns=extra_cols)
    
    # Réordonner selon l'ordre du modèle
    X_aligned = X_aligned[expected_cols]
    
    return X_aligned