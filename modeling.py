
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def entrainer_et_comparer_cv(X, y):
    """
    Phase 4 : Modélisation avec Cross-Validation 5-fold.
    Compare RMSE, R² et MAE sur 4 modèles.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    # Définition des métriques pour la cross-validation
    scoring = {
        'r2': 'r2',
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error'
    }

    results = []

    for name, model in models.items():
        # Exécution de la Cross-Validation 5-fold
        cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)
        
        # On calcule la moyenne des scores (on multiplie par -1 pour MAE et RMSE car sklearn les donne en négatif)
        results.append({
            "Modèle": name,
            "R² (Moyen)": round(cv_results['test_r2'].mean(), 4),
            "MAE Moyen (€)": round(-cv_results['test_mae'].mean(), 2),
            "RMSE Moyen (€)": round(-cv_results['test_rmse'].mean(), 2)
        })

    # Transformation en DataFrame et tri par performance R²
    df_comparison = pd.DataFrame(results).sort_values(by="R² (Moyen)", ascending=False)
    
    return df_comparison