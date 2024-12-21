import librosa
import numpy as np
import torch.nn as nn
from librosa import feature


def extract_mfcc(path):
    audio, sr = librosa.load(path)
    mfccs = librosa.feature.mfcc(audio, sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


def get_feature_vector(file):
    fn_list_i = [feature.chroma_stft, feature.spectral_centroid, feature.spectral_bandwidth, feature.spectral_rolloff]

    fn_list_ii = [feature.rms, feature.zero_crossing_rate]

    y, sr = librosa.load(file, sr=None)

    # Spatial features
    feat_vect_i = [np.mean(funct(y=y, sr=sr)) for funct in fn_list_i]

    # Temporal features
    feat_vect_ii = [np.mean(funct(y=y)) for funct in fn_list_ii]

    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_vector = np.mean(mfccs, axis=1)

    # Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_vector = np.mean(mel_spectrogram, axis=1)

    feature_vector = feat_vect_i + feat_vect_ii + list(mfcc_vector) + list(mel_spectrogram_vector)

    return y, sr, feature_vector


# time-series-array and sampling-rate
# y, sr, vector = get_feature_vector("audio_file.mp3")
# print(y)
# print(sr)
# print(vector)


#  DNN
class AudioVideoFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AudioVideoFeatureExtractor, self).__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.dnn(x)

    # Shape: (B, hidden_dim)
