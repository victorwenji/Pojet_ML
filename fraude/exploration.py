import matplotlib.pyplot as plt
import os

def analyse_classes(df, target='is_fraud'):
    counts = df[target].value_counts()
    ratio = counts.get(1,0) / max(counts.get(0,1), 1)

    # Pie chart
    os.makedirs("static/plots", exist_ok=True)
    plt.figure(figsize=(5,5))
    counts.plot.pie(autopct='%1.1f%%', colors=['lightgreen','red'])
    plt.title("Distribution des classes")
    plt.ylabel("")
    plt.savefig("static/plots/class_distribution.png")
    plt.close()

    # Histogramme
    plt.figure(figsize=(6,4))
    df[target].hist(bins=2, color='lightblue')
    plt.title("Histogramme des classes")
    plt.savefig("static/plots/class_histogram.png")
    plt.close()

    return ratio, "static/plots/class_distribution.png", "static/plots/class_histogram.png"
