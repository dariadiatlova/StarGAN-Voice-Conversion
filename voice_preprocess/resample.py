import shutil
import os

import torchaudio
import torchaudio.functional as F

from tqdm import tqdm


def main(source_path, target_path, resample_rate: int = 22050):
    os.makedirs(target_path, exist_ok=True)
    filenames = os.listdir(source_path)
    for f in tqdm(filenames):
        if ".wav" not in f:
            continue
        wav, sample_rate = torchaudio.load(f"{source_path}/{f}")
        if sample_rate != resample_rate:
            wav = F.resample(wav, sample_rate, resample_rate)
        torchaudio.save(f"{target_path}/{f}", wav, sample_rate=resample_rate)


if __name__ == "__main__":
    source_path = "/root/storage/dasha/data/youtube/hleb/after_vad/48k_super_clean/"
    target_path = "/root/storage/dasha/data/youtube/hleb/after_vad/22k_super_clean/"
    main(source_path, target_path)
