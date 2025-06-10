import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

os.makedirs("outputs", exist_ok=True)

def plot_confusion(y_true, y_pred, title, filename):
    os.makedirs("outputs", exist_ok=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Anomali", "Normal"], yticklabels=["Anomali", "Normal"])
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title(title)
    plt.savefig(f"outputs/{filename}")
    plt.close()