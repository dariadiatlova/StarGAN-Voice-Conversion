import numpy as np
import torch


def get_mel_from_wav(audio, compute_mel):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    melspec, energy = compute_mel(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec, energy
