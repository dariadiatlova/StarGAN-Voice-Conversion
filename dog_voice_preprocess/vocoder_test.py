import librosa
import torch

from audio.compute_mel import ComputeMelEnergy
from audio.tools import get_mel_from_wav
from scipy.io import wavfile


def main(wav_path, output_path, vocoder_path, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80):
    wav, _ = librosa.load(wav_path, sr=sample_rate)
    compute_mel_energy = ComputeMelEnergy(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram, energy = get_mel_from_wav(wav, compute_mel_energy)
    mel_spectrogram = torch.FloatTensor(mel_spectrogram).unsqueeze(0)
    vocoder = torch.jit.load(vocoder_path, map_location="cpu")
    wav_prediction = vocoder(mel_spectrogram.detach().cpu())[0].squeeze(0).detach().cpu().numpy()
    wavfile.write(output_path, sample_rate, wav_prediction)


if __name__ == "__main__":
    wav_path = "/root/storage/dasha/data/dog_dataset/adult_dog_22k/adult_dog_0001.wav"
    output_path = "/root/storage/dasha/data/dog_dataset/reconstructed/rus_voc_22_0001.wav"
    vocoder_path = "/root/storage/dasha/saved_models/hifi_torchscript/rus_1_7M_hifi.pt"
    main(wav_path, output_path, vocoder_path)
