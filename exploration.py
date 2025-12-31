import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def analyser_donnees(df):
    df_paris = df.copy()
    plot_dir = os.path.join('static', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Configuration esthétique
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Distributions (Format 2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    feats = [('prix', 'Prix (€)'), ('surface_habitable', 'Surface (m²)'), 
             ('nb_pieces', 'Pièces'), ('distance_centre', 'Dist. Centre (km)')]
    
    for i, (col, title) in enumerate(feats):
        ax = axes[i//2, i%2]
        sns.histplot(df_paris[col].dropna(), kde=True, ax=ax, color="#4e73df")
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('')

    plt.tight_layout()
    hist_path = 'plots/histograms.png'
    plt.savefig(os.path.join('static', hist_path), dpi=100)
    plt.close()

    # 2. Boxplots (Format compact)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(x=df_paris['prix'], ax=ax1, color='#f6c23e')
    ax1.set_title('Outliers Prix', fontweight='bold')
    sns.boxplot(x=df_paris['surface_habitable'], ax=ax2, color='#1cc88a')
    ax2.set_title('Outliers Surface', fontweight='bold')
    
    box_path = 'plots/boxplots.png'
    plt.savefig(os.path.join('static', box_path), dpi=100)
    plt.close()

    # 3. Calculs et Statistiques (Une seule fois)
    numeric_df = df_paris.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    
    # Tables HTML propres
    stats_html = df_paris.describe().round(2).to_html(classes='table table-sm table-striped')
    corr_prix_html = corr_matrix['prix'].sort_values(ascending=False).head(9).to_frame().to_html(classes='table table-hover table-dark')
    
    # Analyse de santé
    missing = df_paris.isnull().sum()
    report_missing_html = missing[missing > 0].to_frame(name='Manquants').to_html(classes='table table-sm')

    def get_outliers_count(col):
        q1, q3 = df_paris[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        return len(df_paris[(df_paris[col] < q1 - 1.5*iqr) | (df_paris[col] > q3 + 1.5*iqr)])

    outliers_dict = {'Prix': get_outliers_count('prix'), 'Surface': get_outliers_count('surface_habitable')}
    anomalies_dict = {'Distances négatives': len(df_paris[df_paris['distance_transport'] < 0])}

    return stats_html, hist_path, box_path, corr_prix_html, report_missing_html, outliers_dict, anomalies_dict, len(df_paris)