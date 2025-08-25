import torch
from typing import Dict, Tuple
import torch.nn.functional as F

class MultiResSTFTLoss(torch.nn.Module):
    """Multi-resolution STFT loss (spectral convergence + log magnitude)."""
    def __init__(self,
                 fft_sizes=(512, 1024, 2048),
                 hop_sizes=(128, 256, 512),
                 win_lengths=(512, 1024, 2048),
                 window="hann",
                 eps=1e-7,
                 device=None):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.eps = eps
        self.windows = [torch.hann_window(w) for w in win_lengths]
        self.device = device

    def _stft_mag(self, x: torch.Tensor, fft: int, hop: int, win_len: int, window: torch.Tensor):
        x = x.float()
        if self.device is not None and x.device != self.device:
            x = x.to(self.device)
        window = window.to(x.device)
        X = torch.stft(
            x, n_fft=fft, hop_length=hop, win_length=win_len,
            window=window, center=True, return_complex=True
        )
        return torch.abs(X)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        sc_losses, mag_losses = [], []
        for fft, hop, win_len, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths, self.windows):
            x_mag = self._stft_mag(x, fft, hop, win_len, win)
            y_mag = self._stft_mag(y, fft, hop, win_len, win)
            sc = torch.norm(y_mag - x_mag, p='fro') / (torch.norm(y_mag, p='fro') + self.eps)
            mag = F.l1_loss(torch.log(y_mag + self.eps), torch.log(x_mag + self.eps))
            sc_losses.append(sc); mag_losses.append(mag)
        sc_loss = sum(sc_losses) / len(sc_losses)
        mag_loss = sum(mag_losses) / len(mag_losses)
        total = sc_loss + mag_loss
        return total, {"stft_sc": float(sc_loss.detach().cpu()), "stft_mag": float(mag_loss.detach().cpu())}
