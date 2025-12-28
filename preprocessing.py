import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def feature_engineering(df_raw):
    """
    Regroupe les 4 étapes de la Phase 2 en une seule fonction.
    Entrée : DataFrame brut (CSV complet)
    Sortie : DataFrame prêt pour le ML + l'objet Scaler (pour le site web)
    """
    df = df_raw.copy()

    # --- ÉTAPE 0 : Filtrage et Nettoyage de base ---
    # On ne garde que Paris et on nettoie les erreurs évidentes
    df = df[df['ville'].str.lower() == 'paris'].copy()
    if 'distance_transport' in df.columns:
        df['distance_transport'] = df['distance_transport'].abs()
    
    # --- ÉTAPE 1 : Gestion des valeurs manquantes (Imputation) ---
    # Numériques : Médiane
    num_to_impute = ['nb_chambres', 'annee_construction', 'distance_transport', 'score_commerces', 'parking']
    for col in num_to_impute:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # Catégorielles : Mode (valeur la plus fréquente)
    cat_to_impute = ['chauffage', 'classe_energie']
    for col in cat_to_impute:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # --- ÉTAPE 2 : Feature Engineering (Variables composées) ---
    df['surface_par_piece'] = df['surface_habitable'] / df['nb_pieces']
    # On crée une variable combinée pour l'ancienneté et l'énergie
    df['annee_energetique'] = df['annee_construction'].astype(int).astype(str) + "_" + df['classe_energie']

    # --- ÉTAPE 3 : Encodage (One-Hot Encoding) ---
    # Variables à transformer en colonnes 0/1
    cat_cols = ['quartier', 'type_bien', 'etat', 'chauffage', 'classe_energie']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # Suppression des colonnes inutiles pour le calcul
    cols_to_drop = ['ville', 'date_mise_vente', 'annee_energetique'] # annee_energetique est trop précise pour l'encodage ici
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # --- ÉTAPE 4 : Normalisation (Scaling) ---
    # Variables numériques à mettre à l'échelle
    cols_to_scale = [
        'surface_habitable', 'nb_pieces', 'nb_chambres', 'etage', 
        'distance_centre', 'distance_transport', 'score_commerces', 
        'surface_par_piece'
    ]
    
    scaler = StandardScaler()
    # On vérifie que les colonnes existent avant de scaler
    existing_cols = [c for c in cols_to_scale if c in df.columns]
    df[existing_cols] = scaler.fit_transform(df[existing_cols])

    print(f"Pipeline terminé : {df.shape[0]} lignes et {df.shape[1]} colonnes prêtes.")
    return df, scaler