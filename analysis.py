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

# 1. Per (subject, session, algorithm, epoch_length)
for (subject, session, algorithm, epoch_length), group in results.groupby(['subject', 'session', 'algorithm', 'epoch_length']):
    y_true = sum(group['truth'].apply(toi).tolist(), [])
    y_pred = sum(group['prediction'].apply(toi).tolist(), [])
    fname = f"confusion_matrices/cm_subject-{subject}_session-{session}_alg-{algorithm}_epoch-{epoch_length}.png"
    title = f"Confusion Matrix\nSubject: {subject}, Session: {session}, Algorithm: {algorithm}, Epoch: {epoch_length}"
    plot_conf_matrix(y_true, y_pred, title, fname)

# 2. Per (subject, algorithm, epoch_length)
for (subject, algorithm, epoch_length), group in results.groupby(['subject', 'algorithm', 'epoch_length']):
    y_true = sum(group['truth'].apply(toi).tolist(), [])
    y_pred = sum(group['prediction'].apply(toi).tolist(), [])
    fname = f"confusion_matrices/cm_subject-{subject}_alg-{algorithm}_epoch-{epoch_length}.png"
    title = f"Confusion Matrix\nSubject: {subject}, Algorithm: {algorithm}, Epoch: {epoch_length}"
    plot_conf_matrix(y_true, y_pred, title, fname)

# 3. Per (algorithm, epoch_length)
for (algorithm, epoch_length), group in results.groupby(['algorithm', 'epoch_length']):
    y_true = sum(group['truth'].apply(toi).tolist(), [])
    y_pred = sum(group['prediction'].apply(toi).tolist(), [])
    fname = f"confusion_matrices/cm_algorithm-{algorithm}_epoch-{epoch_length}.png"
    title = f"Confusion Matrix\nAlgorithm: {algorithm}, Epoch: {epoch_length}"
    plot_conf_matrix(y_true, y_pred, title, fname)


# Set seaborn style
sns.set(style="whitegrid")

# Create FacetGrid — do NOT set hue here!
g = sns.FacetGrid(
    results,
    col="epoch_length",
    col_wrap=3,
    height=5,
    sharey=True
)

def scatter_with_subjects_and_median(data, **kwargs):
    # Scatterplot with both hue and style inside the function
    sns.scatterplot(
        data=data,
        x="algorithm",
        y="f1",
        hue="session",
        style="subject",
        alpha=0.7,
        s=100,
        **kwargs
    )

    # Plot medians
    medians = data.groupby("algorithm")["f1"].median()
    for i, alg in enumerate(medians.index):
        plt.plot(i, medians[alg], marker='D', color='black', markersize=8, label=None)

    # Deduplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

# Apply the plotting function
g.map_dataframe(scatter_with_subjects_and_median)

# Final styling
g.set_titles(col_template="Epoch Length: {col_name}")
g.set_axis_labels("Algorithm", "F1 Score")
g.tight_layout()
plt.legend()
g.savefig("f1_scatter_per_algorithm_by_epoch_and_subject.png")
plt.close()