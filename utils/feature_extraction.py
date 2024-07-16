import librosa
import numpy as np


def extract_features(audio_path):
    y, sr = librosa.load(audio_path)

    # Ensure the audio is long enough
    if len(y) < sr:  # Check if the audio is less than 1 second
        raise ValueError("Audio file is too short")

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    combined = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
    return np.mean(combined.T, axis=0)
