import glob
import torchaudio
import torchaudio.functional as F

from tqdm import tqdm
from pathlib import Path


def main(source_path, target_path, resample_rate: int = 22050, n_samples=11000):
    Path(target_path).mkdir(parents=True, exist_ok=True)
    filenames = glob.glob(f"{source_path}/**.wav")[:n_samples]
    for f in tqdm(filenames):
        wav, sample_rate = torchaudio.load(f)
        if sample_rate != resample_rate:
            wav = F.resample(wav, sample_rate, resample_rate)
        torchaudio.save(Path(target_path) / Path(f).name, wav, sample_rate=resample_rate)


if __name__ == "__main__":
    source_path = "/root/storage/dasha/data/RUSLAN/"
    target_path = "/root/storage/dasha/data/youtube/hleb/ruslan/22k_natasha_11k/"
    main(source_path, target_path)
