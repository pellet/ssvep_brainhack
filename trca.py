import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def trca(eeg):
    n_channels, n_samples, n_trials = eeg.shape
    S = np.zeros((n_channels, n_channels))

    for i in range(n_trials - 1):
        x1 = eeg[:, :, i] - np.mean(eeg[:, :, i], axis=1, keepdims=True)
        for j in range(i + 1, n_trials):
            x2 = eeg[:, :, j] - np.mean(eeg[:, :, j], axis=1, keepdims=True)
            S += x1 @ x2.T + x2 @ x1.T

    UX = eeg.reshape(n_channels, -1)
    UX = UX - np.mean(UX, axis=1, keepdims=True)
    Q = UX @ UX.T

    eigvals, eigvecs = eigh(S, Q)
    W = eigvecs[:, np.argsort(eigvals)[::-1]]  # descending order
    return W

# Simulate data: 3 channels, 100 samples, 10 trials
np.random.seed(0)
t = np.linspace(0, 1, 100)
signal = np.sin(2 * np.pi * 10 * t)
eeg = np.random.randn(3, 100, 10) * 0.5
eeg[0] += signal[:, np.newaxis]  # Correctly embed signal in all trials


# Run TRCA
W = trca(eeg)

# Project onto first TRCA component
proj = np.array([W[:, 0].T @ eeg[:, :, i] for i in range(10)])

# Plot projections
plt.plot(t, proj.T)
plt.title("TRCA Component 1 (across trials)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
