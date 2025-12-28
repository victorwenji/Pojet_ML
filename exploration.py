import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


def analyser_donnees(df):
    
    df_paris = df.copy()
    
    # Dossier de sauvegarde
    plot_dir = os.path.join('static', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Génération des Histogrammes (Distributions)
    features = ['prix', 'surface_habitable', 'nb_pieces', 'distance_centre']
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(features, 1):
        plt.subplot(2, 2, i)
        sns.histplot(df_paris[col].dropna(), kde=True, color='skyblue')
        plt.title(f'Distribution de : {col}')
    
    hist_path = 'plots/histograms.png'
    plt.savefig(os.path.join('static', hist_path))
    plt.close()

    # 2. Génération des Boxplots (Outliers)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_paris['prix'], color='lightcoral')
    plt.title('Outliers : Prix')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_paris['surface_habitable'], color='lightgreen')
    plt.title('Outliers : Surface Habitable')
    
    box_path = 'plots/boxplots.png'
    plt.savefig(os.path.join('static', box_path))
    plt.close()

    # 3. Matrice de Corrélation
    # On ne garde que les colonnes numériques pour la corrélation
    numeric_df = df_paris.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matrice de Corrélation - Paris')
    
    corr_path = 'plots/correlation_heatmap.png'
    plt.savefig(os.path.join('static', corr_path))
    plt.close()

    # Statistiques descriptives
    stats_html = df_paris.describe().to_html(classes='table table-sm table-hover')
    
    # Extraire spécifiquement la corrélation avec le PRIX pour l'affichage texte
    corr_prix = corr_matrix['prix'].sort_values(ascending=False).to_frame().to_html(classes='table table-dark')
    # Statistiques descriptives
    stats_html = df_paris.describe().to_html(classes='table table-sm table-hover')
    
    
    # --- 2. Statistiques et Corrélations ---
    stats_html = df_paris.describe().to_html(classes='table table-sm')
    numeric_df = df_paris.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    corr_txt = corr_matrix['prix'].sort_values(ascending=False).to_frame().to_html(classes='table table-dark')
    # --- 3. Rapport de Santé (Valeurs manquantes / Outliers) ---
    missing = df_paris.isnull().sum()
    report_missing_html = missing[missing > 0].to_frame(name='Manquants').to_html(classes='table table-warning')

    def count_outliers(col):
        q1 = df_paris[col].quantile(0.25)
        q3 = df_paris[col].quantile(0.75)
        iqr = q3 - q1
        return len(df_paris[(df_paris[col] < q1 - 1.5*iqr) | (df_paris[col] > q3 + 1.5*iqr)])

    outliers_dict = {
        'Prix': count_outliers('prix'),
        'Surface': count_outliers('surface_habitable')
    }

    anomalies_dict = {
        'Distances négatives': len(df_paris[df_paris['distance_transport'] < 0]),
    }

    # IMPORTANT : On retourne TOUT ce dont le dashboard a besoin
    return stats_html, hist_path, box_path, corr_path, corr_txt, report_missing_html, outliers_dict, anomalies_dict, len(df_paris)
    
    #return stats_html, hist_path, box_path,corr_path, corr_prix, len(df_paris)
