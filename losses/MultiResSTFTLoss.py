import torch
from typing import Dict, Tuple
import torch.nn.functional as F

class MultiResSTFTLoss(torch.nn.Module):
    def __init__(self,
                 fft_sizes=(512, 1024, 2048),
                 hop_sizes=(128, 256, 512),
                 win_lengths=(512, 1024, 2048),
                 eps=1e-7,
                 device=None):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.eps = eps
        self.device = device
        self.windows = [torch.hann_window(w) for w in win_lengths]

    def _stft_mag(self, x, fft, hop, win_len, window):
        if self.device is not None and x.device != self.device:
            x = x.to(self.device)
        window = window.to(x.device)
        X = torch.stft(x.float(), n_fft=fft, hop_length=hop, win_length=win_len,
                       window=window, center=True, return_complex=True)
        return torch.abs(X)  # [B, F, T]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        sc_losses, mag_losses = [], []
        for fft, hop, win_len, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths, self.windows):
            x_mag = self._stft_mag(x, fft, hop, win_len, win)
            y_mag = self._stft_mag(y, fft, hop, win_len, win)
            # per-sample norms
            num = (y_mag - x_mag).float().pow(2).sum(dim=(-2, -1)).sqrt()
            den = y_mag.float().pow(2).sum(dim=(-2, -1)).sqrt().clamp_min(self.eps)
            sc = (num / den).mean()

            # log-mag L1
            x_log = torch.log(x_mag.clamp_min(self.eps))
            y_log = torch.log(y_mag.clamp_min(self.eps))
            mag = torch.mean(torch.abs(y_log - x_log))

            sc_losses.append(sc)
            mag_losses.append(mag)

        sc_loss = sum(sc_losses) / len(sc_losses)
        mag_loss = sum(mag_losses) / len(mag_losses)
        total = sc_loss + mag_loss
        return total, {"stft_sc": float(sc_loss.detach().cpu()),
                       "stft_mag": float(mag_loss.detach().cpu())}
