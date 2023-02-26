import os
import math
import torch
import random
import numpy as np

from typing import Tuple, List
from argparse import Namespace
from scipy.io.wavfile import read
from librosa.util import normalize
from torch.utils.data import Dataset
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, clip_val: float = 1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None))


def dynamic_range_decompression(x):
    return np.exp(x)


def dynamic_range_compression_torch(x, clip_val: float = 1e-5):
    return torch.log(torch.clamp(x, min=clip_val))


def dynamic_range_decompression_torch(x):
    return torch.exp(x)


def spectral_normalize_torch(magnitudes: torch.Tensor):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes: torch.Tensor):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def get_mel_spectrogram(y: torch.Tensor, n_fft: int, num_mels: int, sampling_rate: int, hop_size: int,
                        win_size: int, fmin: int, fmax: int, center=False, **_) -> torch.Tensor:
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(win_size).to(y.device)
    pad_value = int((n_fft - hop_size) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad_value, pad_value), mode='reflect').squeeze(1)
    spectrogram = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                             center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    spectrogram = torch.sqrt(spectrogram.pow(2).sum(-1) + 1e-9)
    mel_spectrogram = torch.matmul(mel_basis, spectrogram)
    normalized_mel_spectrogram = spectral_normalize_torch(mel_spectrogram)
    return normalized_mel_spectrogram
