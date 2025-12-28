import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def selection_features(df):
    # 1. Séparation Features (X) et Cible (y)
    X = df.drop(columns=['prix'])
    y = df['prix']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- MÉTHODE 1 : Tree-based Importance (Random Forest) ---
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    importances_tree = pd.Series(model_rf.feature_importances_, index=X.columns)

    # --- MÉTHODE 2 : Permutation Importance (Plus robuste) ---
    perm_importance = permutation_importance(model_rf, X_test, y_test, n_repeats=10, random_state=42)
    importances_perm = pd.Series(perm_importance.importances_mean, index=X.columns)

    # --- MÉTHODE 3 : Corrélation avec la cible ---
    importances_corr = X.corrwith(y).abs()

    # --- AGGRÉGATION ET SÉLECTION DU TOP 15 ---
    # On crée un score combiné (moyenne des rangs)
    df_importance = pd.DataFrame({
        'Tree': importances_tree,
        'Permutation': importances_perm,
        'Correlation': importances_corr
    })
    
    # On normalise les scores pour qu'ils soient comparables
    df_norm = (df_importance - df_importance.min()) / (df_importance.max() - df_importance.min())
    df_norm['Global_Score'] = df_norm.mean(axis=1)
    
    top_15_features = df_norm['Global_Score'].sort_values(ascending=False).head(15).index.tolist()
    
    return top_15_features, df_norm, model_rf, X_train, X_test, y_train, y_test

def comparer_performance(model, X_train, X_test, y_train, y_test, top_15):
    # Score avec TOUTES les features
    score_all = r2_score(y_test, model.predict(X_test))
    
    # Score avec seulement le TOP 15
    model_top = RandomForestRegressor(n_estimators=100, random_state=42)
    model_top.fit(X_train[top_15], y_train)
    score_top = r2_score(y_test, model_top.predict(X_test[top_15]))
    
    return score_all, score_top