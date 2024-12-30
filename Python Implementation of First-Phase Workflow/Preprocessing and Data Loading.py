import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, welch

# Load ECoG data
def load_ecog_data(file_path):
    """Load .mat file and return the ECoG data matrix."""
    mat_data = sio.loadmat(file_path)
    ecog_data = mat_data['TS_DataMat']  # Replace with the actual key for ECoG data
    labels = mat_data['TimeSeries']  # Replace with the actual key for labels
    return ecog_data, labels

# Bandpass filter for noise removal
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Preprocess data
def preprocess_data(ecog_data, fs):
    """Preprocess data by filtering and mean subtraction."""
    preprocessed = []
    for channel in ecog_data:
        # Remove line noise (50 Hz) and apply bandpass filter
        filtered = bandpass_filter(channel, lowcut=0.1, highcut=50, fs=fs)
        # Subtract mean
        filtered = filtered - np.mean(filtered)
        preprocessed.append(filtered)
    return np.array(preprocessed)
