def compute_features(ecog_data, fs):
    """Compute features for each channel."""
    features = []
    for channel in ecog_data:
        # Compute Power Spectral Density (PSD)
        f, Pxx = welch(channel, fs=fs, nperseg=256)

        # Compute power in specific frequency bands
        delta_power = np.trapz(Pxx[(f >= 0.1) & (f <= 4)], f[(f >= 0.1) & (f <= 4)])
        theta_power = np.trapz(Pxx[(f >= 4) & (f <= 8)], f[(f >= 4) & (f <= 8)])
        high_freq_power = np.trapz(Pxx[(f >= 8) & (f <= 30)], f[(f >= 8) & (f <= 30)])

        features.append([delta_power, theta_power, high_freq_power])
    return np.array(features)
