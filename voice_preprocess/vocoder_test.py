import torch

from scipy.io import wavfile
from pathlib import Path

from audio.get_mel import MAX_WAV_VALUE
from vocoder.generator import Generator
from vocoder.stft import TorchSTFT
from vocoder.utils import load_config, load_checkpoint


def main(ruslan_mel_path, hleb_mel_path, output_path, config, device, checkpoint_file, sample_rate=22050):
    ruslan_mel = torch.load(ruslan_mel_path).to(device)
    hleb_mel = torch.load(hleb_mel_path).to(device)
    stft = TorchSTFT(**config).to(device)
    generator = Generator(config).to(device)
    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])
    generator.eval()
    generator.remove_weight_norm()
    for i, mel in enumerate((ruslan_mel, hleb_mel)):
        with torch.no_grad():
            spec, phase = generator(mel)
            y_g_hat = stft.inverse(spec, phase)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            wavfile.write(Path(output_path) / f"{i}_generated.wav", sample_rate, audio)


if __name__ == "__main__":
    ruslan_mel_path = "/root/storage/dasha/data/youtube/hleb/train_paired/mels/0_100.pt"
    hleb_mel_path = "/root/storage/dasha/data/youtube/hleb/train_paired/mels/1_100.pt"
    output_path = "/root/storage/dasha/repos/hleb-vc/voice_preprocess/reconstructed_samples"
    config = load_config("/root/storage/dasha/repos/hleb-vc/vocoder/config.json")
    device = "cuda"
    checkpoint_file = "/root/storage/dasha/repos/iSTFTNet-pytorch/cp_hifigan/g_00975000"
    main(ruslan_mel_path, hleb_mel_path, output_path, config, device, checkpoint_file)
