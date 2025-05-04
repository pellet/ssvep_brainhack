import numpy as np
from scipy import signal
from scipy.linalg import eigh

# MNE functions
from mne import Epochs, find_events
from mne.io import read_raw_edf

# Machine learning
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import metrics

# Constants
FREQ_N = 1024
FREQUENCIES = [9, 10, 12, 15]

def peak_psd(epoch, fs):
    f, Pxx = signal.welch(epoch, fs, nperseg=FREQ_N)
    mask = (f >= 8.5) & (f <= 30.5)
    f_range = f[mask]
    Pxx_range = Pxx[mask]
    peak_freq = f_range[np.argmax(Pxx_range)]
    return peak_freq

def trca(X):  # X shape: (n_trials, n_channels, n_samples)
    n_trials, n_channels, n_samples = X.shape
    S = np.zeros((n_channels, n_channels))

    for i in range(n_trials - 1):
        x1 = X[i] - X[i].mean(axis=1, keepdims=True)
        for j in range(i + 1, n_trials):
            x2 = X[j] - X[j].mean(axis=1, keepdims=True)
            S += x1 @ x2.T + x2 @ x1.T

    UX = X.transpose(1, 0, 2).reshape(n_channels, -1)
    UX = UX - UX.mean(axis=1, keepdims=True)
    Q = UX @ UX.T

    eigvals, eigvecs = eigh(S, Q)
    W = eigvecs[:, np.argsort(eigvals)[::-1]]
    return W

class PSDAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, fs, target_freqs):
        self.fs = fs
        self.target_freqs = target_freqs

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
                peak_freq = peak_psd(epoch[ch], self.fs)
                for freq in self.target_freqs:
                    harmonics = [freq, freq * 2, freq / 2]
                    distances = [abs(peak_freq - h) for h in harmonics]
                    error[freq] += min(distances)
            preds.append(min(error, key=error.get))
        return np.array(preds)

class TRCAClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, fs, target_freqs):
        self.fs = fs
        self.target_freqs = target_freqs
        self.filters_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for label in self.classes_:
            trials = X[y == label]
            W = trca(trials)
            self.filters_[label] = W[:, 0]  # Keep first component
        return self

    def predict(self, X):
        preds = []
        for epoch in X:
            error = {}
            for label, w in self.filters_.items():
                filtered = w.T @ epoch
                peak_freq = peak_psd(filtered, self.fs)
                harmonics = [label, label * 2, label / 2]
                distances = [abs(peak_freq - h) for h in harmonics]
                error[label] = min(distances)
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


# Processing loop
for session in dataset:
    subject = session['subject']
    session_num = session['session']
    edf_file_path = session['path']

    print(f"\nProcessing Subject {subject}, Session {session_num}")
    raw = load_data(edf_file_path)
    print(raw.info)
    fs = int(raw.info['sfreq'])

    events = find_events(raw, stim_channel='stim', verbose=False)

    # Map stimulus order to FREQUENCIES
    for i in range(len(events)):
        events[i][2] = 3 - (i % 4)

    epochs = Epochs(raw, events=events, event_id=[0, 1, 2, 3],
                    tmin=0, tmax=7, baseline=None, preload=True,
                    verbose=False, picks=[1, 2, 3, 4, 5, 6, 7, 8])

    X = epochs.get_data()  # (n_epochs, n_channels, n_samples)
    y = np.array([FREQUENCIES[event[2]] for event in epochs.events])

    # PSDA
    psda = PSDAClassifier(fs=fs, target_freqs=FREQUENCIES)
    psda.fit(X, y)
    y_pred_psda = psda.predict(X)

    # TRCA
    trca_clf = TRCAClassifier(fs=fs, target_freqs=FREQUENCIES)
    trca_clf.fit(X, y)
    y_pred_trca = trca_clf.predict(X)

    # Evaluation
    print("True Labels:   ", y.tolist())
    print("PSDA Predict:  ", y_pred_psda.tolist())
    print("TRCA Predict:  ", y_pred_trca.tolist())

    print(f"PSDA Accuracy: {metrics.accuracy_score(y, y_pred_psda):.3f}")
    print(f"TRCA Accuracy: {metrics.accuracy_score(y, y_pred_trca):.3f}")
