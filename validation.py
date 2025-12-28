# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# # Définition du Pipeline
# pipeline_final = Pipeline([
#     ('scaler', StandardScaler()), # Normalisation
#     ('selector', SelectFromModel(RandomForestRegressor(), max_features=15)), # Top 15 Features
#     ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)) # Meilleur modèle
# ])

# # Entraînement sur 70% des données
# pipeline_final.fit(X_train, y_train)