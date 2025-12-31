import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Remplissage des valeurs manquantes
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64', 'Int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Inconnu")
    
    # Arrondir certaines colonnes float
    round_2_cols = ["revenus_mensuels", "montant_transaction",
                    "montant_total_24h", "distance_trans_precedente", "temps_depuis_derniere_trans",
                    "montant_moyen_30j", "ratio_montant_moyen"]
    for col in round_2_cols:
        if col in df.columns:
            df[col] = df[col].round(2)
    
    # Colonnes entières
    int_cols = ["age_client", "anciennete_compte", "nb_trans_24h", "nb_pays_24h",
                "nb_tentatives_echouees", "auth_3d_secure", "ip_match_pays",
                "appareil_connu", "score_risque_marchand"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: int(round(x)) if pd.notna(x) else 0).astype(np.int64)

    
    # Encoder les colonnes catégorielles spécifiques
    cat_cols = ["type_compte", "statut_professionnel", "region", "type_transaction",
                "categorie_marchand", "pays_transaction", "mode_paiement"]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # Transformer date_transaction en année et mois
    if "date_transaction" in df.columns:
        df["date_transaction"] = pd.to_datetime(df["date_transaction"], errors="coerce")
        df["annee_transaction"] = df["date_transaction"].dt.year.fillna(df["date_transaction"].dt.year.median())
        df["mois_transaction"] = df["date_transaction"].dt.month.fillna(df["date_transaction"].dt.month.median())
        df.drop(columns=["date_transaction"], inplace=True)
    
    return df
