import torchaudio
import torch
import glob

from tqdm import trange
from pathlib import Path

from audio.get_mel import get_mel_spectrogram, MAX_WAV_VALUE


def run(ruslan_voice_dir, hleb_voice_dir, paired_dataset_train_dir, paired_dataset_test_dir, test_dataset_size,
        n_fft=1024, hop_length=256, n_mels=80):
    Path(Path(paired_dataset_train_dir) / "wavs").mkdir(parents=True, exist_ok=True)
    Path(Path(paired_dataset_train_dir) / "mels").mkdir(parents=True, exist_ok=True)
    Path(Path(paired_dataset_test_dir) / "wavs").mkdir(parents=True, exist_ok=True)
    Path(Path(paired_dataset_test_dir) / "mels").mkdir(parents=True, exist_ok=True)
    human_wav_files = glob.glob(f"{ruslan_voice_dir}/**.wav")
    dog_wav_files = glob.glob(f"{hleb_voice_dir}/**.wav")
    dataset_size = min(len(human_wav_files), len(dog_wav_files))
    test_step = dataset_size // test_dataset_size

    for j in trange(dataset_size):
        if j % test_step == 0:
            target_dir = paired_dataset_test_dir
        else:
            target_dir = paired_dataset_train_dir
        ruslan_voice, sr = torchaudio.load(human_wav_files[j])
        hleb_voice, sr = torchaudio.load(dog_wav_files[j])
        audio_size = min(ruslan_voice.shape[1], hleb_voice.shape[1])
        torchaudio.save(f"{target_dir}/wavs/0_{j}.wav", ruslan_voice[:, :audio_size], sr)
        torchaudio.save(f"{target_dir}/wavs/1_{j}.wav", hleb_voice[:, :audio_size], sr)

        ruslan_mel = get_mel_spectrogram(y=ruslan_voice[:, :audio_size], n_fft=n_fft,
                                         num_mels=n_mels, sampling_rate=sr,
                                         hop_size=hop_length, win_size=n_fft, fmin=0, fmax=8000)
        hleb_mel = get_mel_spectrogram(y=hleb_voice[:, :audio_size], n_fft=n_fft,
                                       num_mels=n_mels, sampling_rate=sr,
                                       hop_size=hop_length, win_size=n_fft, fmin=0, fmax=8000)

        torch.save(ruslan_mel, f"{target_dir}/mels/0_{j}.pt")
        torch.save(hleb_mel, f"{target_dir}/mels/1_{j}.pt")


if __name__ == "__main__":
    ruslan_voice_dir = "/root/storage/dasha/data/youtube/hleb/ruslan/22k_natasha_11k/"
    hleb_voice_dir = "/root/storage/dasha/data/natasha/preprocessed_istft/trimmed_wav/"
    paired_dataset_train_dir = "/root/storage/dasha/data/youtube/hleb/train_paired_ruslan_natasha"
    paired_dataset_test_dir = "/root/storage/dasha/data/youtube/hleb/test_paired_ruslan_natasha"
    test_dataset_size = 16
    run(ruslan_voice_dir, hleb_voice_dir, paired_dataset_train_dir, paired_dataset_test_dir, test_dataset_size)
