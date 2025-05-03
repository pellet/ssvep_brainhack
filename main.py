import mne.viz
import numpy as np
import warnings

from scipy import signal

warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt

# MNE functions
from mne import Epochs,find_events
from mne.time_frequency import tfr_morlet
from mne.io import read_raw_edf

# Machine learning functions
from sklearn.pipeline import make_pipeline
from sklearn import metrics

FREQ_N = 1024
FREQUENCIES = [9, 10, 12, 15]

def peak_psd(epoch, fs):
    # Compute the PSD using Welch's method
    f, Pxx = signal.welch(epoch, fs, nperseg=FREQ_N)

    # Limit to the desired frequency range [7, 30] Hz
    mask = (f >= 8.5) & (f <= 30.5)
    f_range = f[mask]
    Pxx_range = Pxx[mask]

    # Find the peak frequency and its corresponding power within the range
    peak_freq = f_range[np.argmax(Pxx_range)]

    return peak_freq

dataset = [
    {
        'subject': 1,
        'session': 1,
        'path': 'data/subject_1_fvep_led_training_1.EDF',
    },
    {
        'subject': 1,
        'session': 2,
        'path': 'data/subject_1_fvep_led_training_2.EDF',
    },
    {
        'subject': 2,
        'session': 1,
        'path': 'data/subject_2_fvep_led_training_1.EDF',
    },
    {
        'subject': 2,
        'session': 2,
        'path': 'data/subject_2_fvep_led_training_2.EDF',
    }
]

for session in dataset:
    subject = session['subject']
    session_num = session['session']
    edf_file_path = session['path']

    print(f"Processing Subject {subject}, Session {session_num}")
    raw = read_raw_edf(edf_file_path, preload=True, verbose=False)
    fs = int(raw.info['sfreq'])

    events = find_events(raw, stim_channel='10', verbose=False)

    classes = []
    for i in range(len(events)):
        if i % 4 == 0:
            events[i][2] = 3
            classes.append(3)
        elif i % 4 == 1:
            events[i][2] = 2
            classes.append(2)
        elif i % 4 == 2:
            events[i][2] = 1
            classes.append(1)
        elif i % 4 == 3:
            events[i][2] = 0
            classes.append(0)

    epochs = Epochs(raw, events=events, event_id=[0, 1, 2, 3],
                    tmin=0, tmax=7, baseline=None, preload=True,
                    verbose=False, picks=[1,2,3,4,5,6,7,8])
    CHANNELS_NUM = epochs.get_data().shape[1]
    EPOCHS_NUM = epochs.get_data().shape[0]

    # PSDA
    # For each channel find the peak between 7-30Hz, then we compute the distance to target frequencies, as well as their closest harmonics (up and down). We choose the closest one.

    y = []
    y_pred = []
    for epoch_id in range(EPOCHS_NUM):
        epoch_target = FREQUENCIES[epochs.events[epoch_id][2]]
        epoch = epochs.get_data()[epoch_id]
        error = {freq: 0.0 for freq in FREQUENCIES}
        for channel in range(CHANNELS_NUM):
            peak_freq = peak_psd(epoch[channel], fs)
            # Accumulate the error for each target frequency considering harmonics
            for freq in FREQUENCIES:
                harmonics = [freq, freq * 2, freq / 2]
                distances = [abs(peak_freq - h) for h in harmonics]

                error[freq] += min(distances)
            # plt.figure()
            # plt.plot(epoch.T)
            # plt.show()
            #
            # plt.figure()
            # plt.psd(np.hanning(len(epoch[6].T))*epoch[6].T, NFFT=FREQ_N, Fs=fs)
            # plt.title('Power Spectral Density of First epoch Wave')
            # plt.xlim(0, 32)
            # plt.show()
        # print(error)
        best_freq = min(error, key=error.get)
        # print(epoch_target, best_freq)
        #
        # exit()

        y.append(epoch_target)
        y_pred.append(best_freq)

    # Print the results
    print("Target : ", y)
    print("Predict: ", y_pred)
    print("Classification Report:\n", metrics.accuracy_score(y, y_pred))

