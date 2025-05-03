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
from sklearn.cross_decomposition import CCA
from mne.simulation import add_noise


edf_file_path = 'data/subject_1_fvep_led_training_1.EDF'
raw = read_raw_edf(edf_file_path, preload=True)

fs = int(raw.info['sfreq'])

def generate_ref(fs, f1, harmonics=1, duration=1.0):
    t = np.arange(0, duration, duration/fs)
    wave = np.zeros_like(t)
    for h in range(1, harmonics + 2):
        print(h)
        wave += signal.chirp(t, f0=f1*h, f1=f1*h, t1=duration, method='linear')/ (h**2)
    return t, wave


# t, wave = generate_ref(fs, 9, 1)
# plt.figure()
# plt.plot(t, wave)
# plt.title('Sine Wave')
# plt.show()
#
# # show PSD of sine wave
# plt.figure()
# plt.psd(np.hanning(len(wave))*wave, NFFT=1024, Fs=fs)
# plt.title('Power Spectral Density of Sine Wave')
# plt.show()
#
# exit()

events = find_events(raw, stim_channel='10')


classes = []
for i in range(len(events)):
    if i % 4 == 0:
        events[i][2] = 0
        classes.append(0)
    elif i % 4 == 1:
        events[i][2] = 1
        classes.append(1)
    elif i % 4 == 2:
        events[i][2] = 2
        classes.append(2)
    elif i % 4 == 3:
        events[i][2] = 3
        classes.append(3)

epochs = Epochs(raw, events=events, event_id=[0, 1, 2, 3],
                tmin=0, tmax=7, baseline=None, preload=True,
                verbose=False, picks=[1,2,3,4,5,6,7,8])
first_epoch = epochs.get_data()[1][6]

plt.figure()
plt.plot(first_epoch)
plt.show()

plt.figure()
plt.psd(np.hanning(len(first_epoch))*first_epoch, NFFT=2048, Fs=fs)
plt.title('Power Spectral Density of Sine Wave')
plt.xlim(0, 20)
plt.show()