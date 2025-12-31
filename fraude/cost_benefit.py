import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

def analyze_cost_benefit(y_true, y_pred, y_prob, threshold=0.5, 
                         cost_fp=50, cost_fn=500, benefit_tp=500):
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculs financiers
    total_cost_fp = fp * cost_fp
    total_cost_fn = fn * cost_fn
    total_benefit_tp = tp * benefit_tp
    
    net_benefit = total_benefit_tp - total_cost_fp - total_cost_fn
    
    # Optimisation du seuil pour maximiser le bénéfice net
    thresholds = np.linspace(0.1, 0.9, 50)
    net_benefits = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        cm_thresh = confusion_matrix(y_true, y_pred_thresh)
        tn_t, fp_t, fn_t, tp_t = cm_thresh.ravel()
        
        benefit_t = (tp_t * benefit_tp) - (fp_t * cost_fp) - (fn_t * cost_fn)
        net_benefits.append(benefit_t)
    
    optimal_idx = np.argmax(net_benefits)
    optimal_threshold = thresholds[optimal_idx]
    max_benefit = net_benefits[optimal_idx]
    
    # Résultats
    results = {
        'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)},
        'costs': {
            'cost_fp_total': float(total_cost_fp),
            'cost_fn_total': float(total_cost_fn),
            'benefit_tp_total': float(total_benefit_tp),
            'net_benefit': float(net_benefit)
        },
        'optimal': {
            'threshold': float(optimal_threshold),
            'max_benefit': float(max_benefit)
        }
    }
    
    # Graphiques
    generate_cost_benefit_plots(thresholds, net_benefits, optimal_threshold, 
                                tp, fp, fn, tn, cost_fp, cost_fn, benefit_tp)
    
    return results

def generate_cost_benefit_plots(thresholds, net_benefits, optimal_threshold,
                                tp, fp, fn, tn, cost_fp, cost_fn, benefit_tp):
    
    os.makedirs("static/plots", exist_ok=True)
    
    # 1. Bénéfice net vs Seuil
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(thresholds, net_benefits, linewidth=2, color='#2ecc71')
    ax1.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Seuil optimal = {optimal_threshold:.3f}')
    ax1.scatter([optimal_threshold], [max(net_benefits)], color='red', s=100, zorder=5)
    ax1.set_xlabel('Seuil de décision', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Bénéfice Net (€)', fontsize=11, fontweight='bold')
    ax1.set_title('Optimisation du Bénéfice Net', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Répartition des coûts/bénéfices
    categories = ['Bénéfice\n(TP)', 'Coût\n(FP)', 'Coût\n(FN)']
    values = [tp * benefit_tp, fp * cost_fp, fn * cost_fn]
    colors_bar = ['#2ecc71', '#e74c3c', '#e67e22']
    
    bars = ax2.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Montant (€)', fontsize=11, fontweight='bold')
    ax2.set_title('Analyse Coûts vs Bénéfices', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Ajouter valeurs sur barres
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,.0f}€', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("static/plots/cost_benefit_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Matrice de confusion avec coûts
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm_display = np.array([[tn, fp], [fn, tp]])
    labels_display = np.array([
        [f'TN: {tn}\n(Correct)', f'FP: {fp}\n(-{cost_fp}€)'],
        [f'FN: {fn}\n(-{cost_fn}€)', f'TP: {tp}\n(+{benefit_tp}€)']
    ])
    
    sns.heatmap(cm_display, annot=labels_display, fmt='', cmap='RdYlGn_r', 
                cbar=True, linewidths=2, linecolor='black',
                xticklabels=['Prédit: Légitime', 'Prédit: Fraude'],
                yticklabels=['Réel: Légitime', 'Réel: Fraude'],
                ax=ax)
    
    ax.set_title('Matrice de Confusion avec Impact Financier', 
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig("static/plots/confusion_matrix_cost.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Graphiques coût-bénéfice générés!")