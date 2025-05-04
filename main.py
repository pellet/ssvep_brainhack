import numpy as np
from scipy import signal
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# MNE functions
from mne import Epochs, find_events
from mne.io import read_raw_edf

# Machine learning
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# Constants
FREQ_N = [1024, 257, 769]
FREQUENCIES = [9, 10, 12, 15]
LENGTH_S = [7, 1, 3]

def peak_psd(epoch, fs, nperseg=FREQ_N[0]):
    f, Pxx = signal.welch(epoch, fs, nperseg=nperseg)
    mask = (f >= 8.5) & (f <= 30.5)
    f_range = f[mask]
    Pxx_range = Pxx[mask]
    peak_freq = f_range[np.argmax(Pxx_range)]
    return peak_freq

class PSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, fs, target_freqs, length=7):
        self.fs = fs
        self.target_freqs = target_freqs
        self.length = length

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        preds = []
        for epoch in X:
            error = {freq: 0.0 for freq in self.target_freqs}
            for ch in range(epoch.shape[0]):
                nperseg = FREQ_N[0]
                if self.length == LENGTH_S[1]:
                    nperseg = FREQ_N[1]
                elif self.length == LENGTH_S[2]:
                    nperseg = FREQ_N[2]
                peak_freq = peak_psd(epoch[ch], self.fs, nperseg=nperseg)
                for freq in self.target_freqs:
                    harmonics = [freq, freq * 2, freq / 2]
                    distances = [abs(peak_freq - h) for h in harmonics]
                    error[freq] += min(distances)
            preds.append(min(error, key=error.get))
        return np.array(preds)

# Dataset definition
dataset = [
    {'subject': 1, 'session': 1, 'path': 'data/subject_1_fvep_led_training_1.EDF'},
    {'subject': 1, 'session': 2, 'path': 'data/subject_1_fvep_led_training_2.EDF'},
    {'subject': 2, 'session': 1, 'path': 'data/subject_2_fvep_led_training_1.EDF'},
    {'subject': 2, 'session': 2, 'path': 'data/subject_2_fvep_led_training_2.EDF'}
]


def load_data(edf_file_path):
    # Set appropriate channel types for non-EEG channels
    raw = read_raw_edf(edf_file_path, preload=True, verbose=False, stim_channel=9, misc=[0, 10])
    # Assign the correct channel names
    new_channel_names = ['sample time', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'stim',
                         'lda classification']
    raw.rename_channels(dict(zip(raw.ch_names, new_channel_names)))

    raw.set_montage('standard_1020')
    return raw


# Create confusion matrices
def plot_confusion_matrix(y_true, y_pred, title, frequencies):
    cm = confusion_matrix(y_true, y_pred, labels=frequencies)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=frequencies, yticklabels=frequencies)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} Confusion Matrix')
    plt.tight_layout()
    return plt

# `subject;session;algorithm;truth;prediction;epoch_length`

results = []
for length in LENGTH_S:
    # Processing loop
    for session in dataset:
        subject = session['subject']
        session_num = session['session']
        edf_file_path = session['path']

        print(f"\nProcessing Subject {subject}, Session {session_num}")
        raw = load_data(edf_file_path)
        fs = int(raw.info['sfreq'])

        events = find_events(raw, stim_channel='stim', verbose=False)

        # Map stimulus order to FREQUENCIES
        for i in range(len(events)):
            events[i][2] = 3 - (i % 4)

        event_id = {'9 Hz': 3, '10 Hz': 2, '12 Hz': 1, '15 Hz': 0}
        epochs = Epochs(raw, events=events, event_id=event_id,
                        tmin=0, tmax=length, baseline=None, preload=True,
                        verbose=False, picks=['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'])

        X = epochs.get_data()  # (n_epochs, n_channels, n_samples)
        y = np.array([FREQUENCIES[event[2]] for event in epochs.events])
        n_samples = X.shape[0]

        # PSDA: train and test on full
        psda = PSDAClassifier(fs=fs, target_freqs=FREQUENCIES, length=length)
        psda.fit(X, y)
        y_pred_psda = psda.predict(X)

        # Save results
        results.append({
            'subject': subject,
            'session': session_num,
            'algorithm': 'PSDA',
            'truth': ','.join(map(str, y.tolist())),
            'prediction': ','.join(map(str, y_pred_psda.tolist())),
            'epoch_length': length
        })

        # Evaluation
        print("True Labels (all):   ", y.tolist())
        print("PSDA Predict (all):  ", y_pred_psda.tolist())
        print("PSDA Accuracy (all): ", metrics.accuracy_score(y, y_pred_psda))

        # Plot confusion matrices for both classifiers
        # psda_cm_plot = plot_confusion_matrix(y, y_pred_psda, f'PSDA - subject:{subject}, session:{session_num}', FREQUENCIES)
        # Display plots in PyCharm's scientific view
        # plt.show()

results_df = pd.DataFrame(results)
results_df.to_csv('results_psda.csv', index=False)