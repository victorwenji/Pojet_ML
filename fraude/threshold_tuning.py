import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy as np
import os

def threshold_tuning(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    os.makedirs("static/plots", exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(thresholds, f1_scores[:-1], marker='o')
    plt.xlabel("Seuil")
    plt.ylabel("F1-score")
    plt.title("F1-score vs Threshold")
    plt.grid(True)
    plt.savefig("static/plots/f1_vs_threshold.png")
    plt.close()

    return float(best_threshold)
