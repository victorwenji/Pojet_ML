import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from preprocessing import preprocess_data, get_pipeline

# Charger les données
df = pd.read_csv('data/immobilier_dataset.csv')
df = preprocess_data(df)

# Séparer Features et Cible
X = df.drop('prix', axis=1)
y = df['prix']

# Définir les colonnes par type
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du Pipeline complet avec un modèle Random Forest
full_pipeline = Pipeline(steps=[
    ('preprocessor', get_pipeline(numeric_features, categorical_features)),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Optimisation avec GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
}

print("Entraînement en cours...")
grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(f"Meilleur score R² : {r2_score(y_test, y_pred):.4f}")
print(f"Erreur moyenne (MAE) : {mean_absolute_error(y_test, y_pred):.2f} €")

# Sauvegarde du modèle
joblib.dump(best_model, 'models/model_paris.joblib')
print("Modèle sauvegardé dans models/model_paris.joblib")