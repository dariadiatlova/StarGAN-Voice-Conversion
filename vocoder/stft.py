import torch
import numpy as np

from typing import Tuple, Optional
from scipy.signal import get_window


class TorchSTFT(torch.nn.Module):
    def __init__(self, istft_filter_length: int, istft_hop_length: int, window: str = 'hann', **_):
        super().__init__()
        self.hop_length = istft_hop_length
        self.win_length = istft_filter_length
        self.filter_length = istft_filter_length
        self.window = torch.from_numpy(get_window(window, istft_filter_length, fftbins=True).astype(np.float32))

    def transform(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.complex]:
        forward_transform = torch.stft(input_data, self.filter_length, self.hop_length,
                                       self.win_length, window=self.window, return_complex=True)
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude: torch.Tensor, phase: torch.complex) -> torch.Tensor:
        inverse_transform = torch.istft(magnitude * torch.exp(phase * 1j), self.filter_length,
                                        self.hop_length, self.win_length, window=self.window.to(magnitude.device))
        return inverse_transform.unsqueeze(-2)  # unsqueeze to stay consistent with conv_transpose1d implementation

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        magnitude, phase = self.transform(input_data)
        reconstruction = self.inverse(magnitude, phase)
        return reconstruction
