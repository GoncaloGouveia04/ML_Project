import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc_curve(data, candidate_counts):
    
    plt.figure(figsize=(11, 6))

    sns.lineplot(
        data=data,
        x="candidates",
        y="ROC-AUC",
        hue="model",
        marker="o"
    )

    plt.xlabel("Number of hyperparameter candidates")
    plt.ylabel("Best cross-validation ROC-AUC")
    plt.title("Hyperparameter tuning: ROC-AUC vs number of candidates")
    plt.xticks(candidate_counts)
    plt.grid(alpha=0.3)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()