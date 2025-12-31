import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os

from fraude.baseline import train_baseline_model, evaluate_model as eval_baseline
from fraude.class_weight import train_rf_class_weight, evaluate_model as eval_cw
from fraude.smote_model import train_smote, evaluate_model as eval_smote

def compare_all_strategies(X_train, X_test, y_train, y_test):

    results = []
    
    print("🔹 Entraînement Baseline...")
    model_baseline = train_baseline_model(X_train, y_train)
    metrics_baseline = eval_baseline(model_baseline, X_test, y_test)
    metrics_baseline['strategie'] = 'Baseline'
    results.append(metrics_baseline)
    joblib.dump(model_baseline, "models/baseline.pkl")
    
    print("🔹 Entraînement Class Weight...")
    model_cw = train_rf_class_weight(X_train, y_train)
    metrics_cw = eval_cw(model_cw, X_test, y_test)
    metrics_cw['strategie'] = 'Class Weight'
    results.append(metrics_cw)
    joblib.dump(model_cw, "models/class_weight.pkl")
    
    print("🔹 Entraînement SMOTE...")
    model_smote = train_smote(X_train, y_train)
    metrics_smote = eval_smote(model_smote, X_test, y_test)
    metrics_smote['strategie'] = 'SMOTE'
    results.append(metrics_smote)
    joblib.dump(model_smote, "models/smote.pkl")
    
    # Créer DataFrame comparatif
    df_comp = pd.DataFrame(results)
    df_comp = df_comp[['strategie', 'precision', 'recall', 'f1', 'roc_auc']]
    
    # Sauvegarder en CSV
    os.makedirs("static/reports", exist_ok=True)
    df_comp.to_csv("static/reports/comparison.csv", index=False)
    
    # Générer graphiques
    generate_comparison_plots(df_comp)
    
    return df_comp

def generate_comparison_plots(df_comp):
    """
    Génère le graphique comparatif des stratégies
    """
    os.makedirs("static/plots", exist_ok=True)
    
    # Graphique en barres groupées
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, strategy in enumerate(df_comp['strategie']):
        values = df_comp[df_comp['strategie'] == strategy][metrics].values[0]
        offset = (i - 1) * width
        ax.bar(x + offset, values, width, label=strategy, alpha=0.8)
    
    ax.set_xlabel('Métriques', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des Stratégies de Détection de Fraude', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Ajouter les valeurs sur les barres
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
    
    plt.tight_layout()
    plt.savefig("static/plots/strategie_comparaison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Graphique radar
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    categories = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, strategy in enumerate(df_comp['strategie']):
        values = df_comp[df_comp['strategie'] == strategy][metrics].values[0].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Comparaison Radar des Stratégies', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("static/plots/strategie_radar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Graphiques de comparaison générés avec succès!")