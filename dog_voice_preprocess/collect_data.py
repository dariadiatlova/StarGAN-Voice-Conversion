import os
import torchaudio
import torch
import numpy as np

from tqdm import trange

from audio.compute_mel import ComputeMelEnergy
from audio.tools import get_mel_from_wav


def run(human_voice_dir, dog_voice_dir, paired_dataset_train_dir, paired_dataset_test_dir, test_dataset_size,
        sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    os.makedirs(paired_dataset_train_dir, exist_ok=True)
    os.makedirs(paired_dataset_test_dir, exist_ok=True)
    os.makedirs(f"{paired_dataset_train_dir}/wavs", exist_ok=True)
    os.makedirs(f"{paired_dataset_train_dir}/mels", exist_ok=True)
    os.makedirs(f"{paired_dataset_test_dir}/wavs", exist_ok=True)
    os.makedirs(f"{paired_dataset_test_dir}/mels", exist_ok=True)
    human_wav_files = [i for i in os.listdir(human_voice_dir) if "wav" in i]
    dog_wav_files = os.listdir(dog_voice_dir)
    dog_wav_files.extend(dog_wav_files)
    dog_wav_files.extend(dog_wav_files)
    dataset_size = min(len(human_wav_files), len(dog_wav_files))
    test_step = dataset_size // test_dataset_size
    compute_mel_energy = ComputeMelEnergy(sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
                                          n_mels=n_mels)
    for j in trange(dataset_size):
        if j % test_step == 0:
            target_dir = paired_dataset_test_dir
        else:
            target_dir = paired_dataset_train_dir
        human_voice, sr = torchaudio.load(f"{human_voice_dir}/{human_wav_files[j]}")
        dog_voice, sr = torchaudio.load(f"{dog_voice_dir}/{dog_wav_files[j]}")
        audio_size = min(human_voice.shape[1], dog_voice.shape[1])
        torchaudio.save(f"{target_dir}/wavs/0_{j}.wav", human_voice[:, :audio_size], sr)
        torchaudio.save(f"{target_dir}/wavs/1_{j}.wav", dog_voice[:, :audio_size], sr)
        human_mel = get_mel_from_wav(human_voice[:, :audio_size], compute_mel_energy)[0][0, :, :].T
        dog_mel = get_mel_from_wav(dog_voice[:, :audio_size], compute_mel_energy)[0][0, :, :].T
        assert len(human_mel.shape) == len(dog_mel.shape) == 2, f"Mel shapes are: {human_mel.shape}, {dog_mel.shape}"
        np.save(f"{target_dir}/mels/0_{j}.npy", human_mel)
        np.save(f"{target_dir}/mels/1_{j}.npy", dog_mel)


if __name__ == "__main__":
    human_voice_dir = "/root/storage/dasha/data/natasha/aligned_corpus"
    dog_voice_dir = "/root/storage/dasha/data/dog_dataset/adult_dog_22k"
    paired_dataset_train_dir = "/root/storage/dasha/data/dog_dataset/train_human_paired"
    paired_dataset_test_dir = "/root/storage/dasha/data/dog_dataset/test_human_paired"
    test_dataset_size = 64
    run(human_voice_dir, dog_voice_dir, paired_dataset_train_dir, paired_dataset_test_dir, test_dataset_size)
