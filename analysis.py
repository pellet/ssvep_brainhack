import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

def toi(x):
    return list(map(int, x.split(",")))


def compute_metrics(row):
    y_true, y_pred = toi(row.truth), toi(row.prediction)
    return pd.Series({
        'precision': metrics.precision_score(y_true, y_pred, average='macro'),
        'recall': metrics.recall_score(y_true, y_pred, average='macro'),
        'accuracy': metrics.accuracy_score(y_true, y_pred),
        'f1': metrics.f1_score(y_true, y_pred, average='macro')
    })


results = pd.read_csv("results_psda.csv")

results[['precision', 'recall', 'accuracy', 'f1']] = results.apply(compute_metrics, axis=1)

# Ensure output directory exists
os.makedirs("confusion_matrices", exist_ok=True)

def plot_conf_matrix(y_true, y_pred, title, filename):
    cm = metrics.confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true + y_pred)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 1. Per (subject, session, algorithm)
for (subject, session, algorithm), group in results.groupby(['subject', 'session', 'algorithm']):
    y_true = sum(group['truth'].apply(toi).tolist(), [])
    y_pred = sum(group['prediction'].apply(toi).tolist(), [])
    fname = f"confusion_matrices/cm_subject-{subject}_session-{session}_alg-{algorithm}.png"
    title = f"Confusion Matrix\nSubject: {subject}, Session: {session}, Algorithm: {algorithm}"
    plot_conf_matrix(y_true, y_pred, title, fname)

# 2. Per (subject, algorithm)
for (subject, algorithm), group in results.groupby(['subject', 'algorithm']):
    y_true = sum(group['truth'].apply(toi).tolist(), [])
    y_pred = sum(group['prediction'].apply(toi).tolist(), [])
    fname = f"confusion_matrices/cm_subject-{subject}_alg-{algorithm}.png"
    title = f"Confusion Matrix\nSubject: {subject}, Algorithm: {algorithm}"
    plot_conf_matrix(y_true, y_pred, title, fname)

# 3. Per algorithm (overall)
for algorithm, group in results.groupby('algorithm'):
    y_true = sum(group['truth'].apply(toi).tolist(), [])
    y_pred = sum(group['prediction'].apply(toi).tolist(), [])
    fname = f"confusion_matrices/cm_algorithm-{algorithm}.png"
    title = f"Confusion Matrix\nAlgorithm: {algorithm}"
    plot_conf_matrix(y_true, y_pred, title, fname)


# 4. Plot accuracy per algorithm
plt.figure(figsize=(10, 6))

# Scatter plot of accuracy per (session, algorithm)
sns.stripplot(
    data=results,
    x="algorithm",
    y="accuracy",
    hue="session",
    jitter=True,
    dodge=True,
    alpha=0.7
)

# Plot medians per algorithm
medians = results.groupby('algorithm')['accuracy'].median()
for i, alg in enumerate(medians.index):
    plt.plot(i, medians[alg], marker='D', color='black', markersize=8, label=None if i else 'Median')

plt.title("Accuracy per Algorithm per Session")
plt.ylabel("Accuracy")
plt.xlabel("Algorithm")
plt.legend(title="Session", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("accuracy_scatter_per_algorithm.png")
plt.close()