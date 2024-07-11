import librosa
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    combined = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
    return np.mean(combined.T, axis=0)
