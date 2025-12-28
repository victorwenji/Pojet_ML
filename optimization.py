from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import time

def optimiser_modele(X_train, y_train, X_test, y_test):
    """
    Phase 5 : Recherche des meilleurs hyperparamètres (Tuning).
    """
    # 1. Modèle de base (réglages par défaut)
    base_model = GradientBoostingRegressor(random_state=42)
    base_model.fit(X_train, y_train)
    y_pred_base = base_model.predict(X_test)
    score_base = r2_score(y_test, y_pred_base)
    mae_base = mean_absolute_error(y_test, y_pred_base)

    # 2. Définition de la grille d'hyperparamètres (les 4 clés)
    param_grid = {
        'n_estimators': [100, 200],      # Nombre d'arbres
        'learning_rate': [0.05, 0.1],    # Vitesse d'apprentissage
        'max_depth': [3, 5],             # Profondeur des arbres
        'min_samples_split': [2, 5]      # Échantillons min. pour diviser
    }

    # 3. Exécution du GridSearchCV (Validation croisée intégrée)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1, # Utilise tous les processeurs
        verbose=1
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    duration = round(time.time() - start_time, 2)

    # 4. Meilleur modèle après optimisation
    best_model = grid_search.best_estimator_
    y_pred_opti = best_model.predict(X_test)
    score_opti = r2_score(y_test, y_pred_opti)
    mae_opti = mean_absolute_error(y_test, y_pred_opti)

    return {
        "best_params": grid_search.best_params_,
        "score_avant": round(score_base, 4),
        "score_apres": round(score_opti, 4),
        "mae_avant": round(mae_base, 0),
        "mae_apres": round(mae_opti, 0),
        "gain_r2": round(score_opti - score_base, 4),
        "duree": duration,
        "modele_final": best_model
    }