import torch
import torch.nn as nn

class MultiResSTFTLoss(nn.Module):
    """
    iSTFTNet-style multi-resolution STFT loss:
      total = mean_i [ SC_i + L1( log|X_i| - log|Y_i| ) ]

    Where SC_i = || |Y_i| - |X_i| ||_F / ( || |Y_i| ||_F + eps )
    """
    def __init__(
        self,
        fft_sizes=(1024, 2048, 512, 256, 128),
        hop_sizes=(256,  512,  128,  64,  32),
        win_lengths=(1024, 2048, 512, 256, 128),
        window="hann",
        eps=1e-7,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes   = tuple(int(x) for x in fft_sizes)
        self.hop_sizes   = tuple(int(x) for x in hop_sizes)
        self.win_lengths = tuple(int(x) for x in win_lengths)
        self.eps = float(eps)

        wins = []
        for w in self.win_lengths:
            if window != "hann":
                raise ValueError("Only 'hann' window is supported.")
            wins.append(torch.hann_window(w))
        for i, w in enumerate(wins):
            self.register_buffer(f"_win_{i}", w, persistent=False)

    def _get_window(self, i: int) -> torch.Tensor:
        return getattr(self, f"_win_{i}")

    @staticmethod
    def _ensure_bt(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:     # [T] -> [1,T]
            x = x.unsqueeze(0)
        if x.dim() == 3 and x.size(1) == 1:  # [B,1,T] -> [B,T]
            x = x.squeeze(1)
        assert x.dim() == 2, f"Expected [B,T], got {tuple(x.shape)}"
        return x

    def _stft_mag(self, x: torch.Tensor, n_fft: int, hop: int, win_len: int, window: torch.Tensor):
        X = torch.stft(
            x.float(),
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_len,
            window=window,
            center=True,
            return_complex=True,
        )
        return torch.abs(X)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        x = self._ensure_bt(pred)
        y = self._ensure_bt(target)

        sc_losses, mag_losses = [], []
        for i, (fft, hop, wlen) in enumerate(zip(self.fft_sizes, self.hop_sizes, self.win_lengths)):
            win = self._get_window(i).to(x.device, dtype=x.dtype)

            x_mag = self._stft_mag(x, fft, hop, wlen, win)
            y_mag = self._stft_mag(y, fft, hop, wlen, win)

            num = (y_mag - x_mag).pow(2).sum(dim=(-2, -1)).sqrt()
            den = y_mag.pow(2).sum(dim=(-2, -1)).sqrt().clamp_min(self.eps)
            sc = (num / den).mean()

            x_log = torch.log(x_mag.clamp_min(self.eps))
            y_log = torch.log(y_mag.clamp_min(self.eps))
            mag = torch.mean(torch.abs(y_log - x_log))

            sc_losses.append(sc)
            mag_losses.append(mag)

        sc_loss  = sum(sc_losses) / len(sc_losses)
        mag_loss = sum(mag_losses) / len(mag_losses)
        total    = sc_loss + mag_loss
        return total, {"stft_sc": float(sc_loss.detach().cpu()),
                       "stft_mag": float(mag_loss.detach().cpu())}