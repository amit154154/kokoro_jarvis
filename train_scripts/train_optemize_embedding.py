#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, json, math
from typing import  Dict, Any

from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from datasets.LibriTTSSingleSpeakerDataset import (
    LibriTTSSingleSpeakerDataset as LTS_Spk_Ds,
    collate_fn as lts_collate_fn,
)
from utills import *

from kokoro.model import KModel
from kokoro import KPipeline

SR = 24000

class KokoroEmbedOptimModule(pl.LightningModule):
    def __init__(
        self,
        mode: str,
        base_packs: List[torch.FloatTensor],
        voice_names: List[str],
        steps_per_epoch: int,
        max_epochs: int,

        # optimization
        logits_lr: float = 1e-2,       # high LR for mixture logits
        emb_lr: float = 2e-4,          # LR if optimizing full embedding
        weight_decay: float = 0.0,

        # scheduler
        scheduler: str = "onecycle",   # onecycle | cosine | cosine_warmup | plateau | none
        warmup_frac: float = 0.1,

        # softmax / exploration
        temp_start: float = 1.5,
        temp_end: float = 0.3,
        gumbel: bool = False,
        entropy_coeff: float = -0.001,  # encourages spread early (negative * H)

        # loss knobs
        si_loss: bool = True,           # scale-invariant (gain match) before MR-STFT
        loss_log10: bool = True,        # use log10 instead of ln for mag loss
        l1_weight: float = 0.05,
        stft_device: str = "device",    # 'device' | 'cpu'

        # logging
        log_every: int = 5,
        demo_log_every: int = 15,
        demo_text: str = "Hi! This is an optimized Kokoro voice embedding.",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["base_packs", "voice_names"])

        # Model (frozen)
        self.model = KModel(repo_id="hexgrad/Kokoro-82M")
        for p in self.model.parameters():
            p.requires_grad = False

        self.voice_names = voice_names

        # Pre-stack base packs as buffer (Lightning moves buffers to device)
        if mode == "mixture":
            base = torch.stack(base_packs, dim=0)  # [N,T,1,D] on CPU
            self.register_buffer("base_packs_buf", base, persistent=False)
        else:
            self.base_packs_buf = None

        # STFT windows as BUFFERS; placed onto chosen device in on_train_start
        self.register_buffer("win_512",  torch.hann_window(512),  persistent=False)
        self.register_buffer("win_1024", torch.hann_window(1024), persistent=False)
        self.register_buffer("win_2048", torch.hann_window(2048), persistent=False)

        # remember where to compute loss
        self._stft_device_choice = stft_device
        self._stft_dev = None  # set in on_train_start

        # Trainables
        if mode == "mixture":
            N = len(voice_names)
            self.logits = torch.nn.Parameter(torch.zeros(N, dtype=torch.float32))
            self.emb_param = None
        else:
            init = average_pack(base_packs)  # [T,1,D]
            self.emb_param = torch.nn.Parameter(init.clone().detach().to(torch.float32))
            self.logits = None

        # quiet G2P pipe (we pass model at call time)
        self._demo_pipe = KPipeline(lang_code="a", model=False)

    # -------- helpers --------
    def _progress(self) -> float:
        total = max(1, self.hparams.steps_per_epoch * self.hparams.max_epochs)
        return min(1.0, float(self.global_step) / float(total))

    def _current_temp(self) -> float:
        p = self._progress()
        return self.hparams.temp_end + (self.hparams.temp_start - self.hparams.temp_end) * (1.0 - p)

    def _current_weights(self) -> torch.Tensor:
        temp = max(1e-4, self._current_temp())
        if self.hparams.gumbel:
            return F.gumbel_softmax(self.logits, tau=temp, hard=False, dim=-1)
        else:
            return F.softmax(self.logits / temp, dim=-1)

    def _current_pack(self) -> torch.FloatTensor:
        if self.hparams.mode == "mixture":
            w = self._current_weights()        # [N]
            mixed = torch.einsum("n,ntcd->tcd", w, self.base_packs_buf)
            return mixed
        else:
            return self.emb_param

    @staticmethod
    def _pad_and_crop(preds: List[torch.Tensor], wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(x.shape[-1] for x in preds)
        preds_padded = torch.stack([F.pad(x, (0, max_len - x.shape[-1])) for x in preds], dim=0).squeeze(1)
        L = min(wav.shape[-1], preds_padded.shape[-1])
        return preds_padded[..., :L], wav[..., :L]

    @staticmethod
    def _sanitize_wave(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-2.0, 2.0)

    def _stft_mag(self, x: torch.Tensor, n_fft: int, hop: int, win_len: int, window: torch.Tensor, use_log10: bool):
        if x.device != self._stft_dev:
            x = x.to(self._stft_dev)
        X = torch.stft(x.float(), n_fft=n_fft, hop_length=hop, win_length=win_len,
                       window=window, center=True, return_complex=True)
        mag = torch.abs(X).clamp_min(1e-7)
        return torch.log10(mag) if use_log10 else torch.log(mag)

    def _mrstft_loss(self, x: torch.Tensor, y: torch.Tensor, use_log10=True):
        if x.device != self._stft_dev: x = x.to(self._stft_dev)
        if y.device != self._stft_dev: y = y.to(self._stft_dev)

        cfgs = [
            (512, 128, 512, self.win_512),
            (1024, 256, 1024, self.win_1024),
            (2048, 512, 2048, self.win_2048),
        ]
        sc_losses, mag_losses = [], []
        for nfft, hop, win, win_t in cfgs:
            x_mag = self._stft_mag(x, nfft, hop, win, win_t, use_log10)
            y_mag = self._stft_mag(y, nfft, hop, win, win_t, use_log10)
            num = (y_mag - x_mag).pow(2).sum(dim=(-2, -1)).sqrt()
            den = y_mag.pow(2).sum(dim=(-2, -1)).sqrt().clamp_min(1e-7)
            sc = (num / den).mean()
            mag = torch.mean(torch.abs(y_mag - x_mag))
            sc_losses.append(sc)
            mag_losses.append(mag)
        return sum(sc_losses) / len(sc_losses), sum(mag_losses) / len(mag_losses)

    def _align_gain(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        num = (pred * tgt).sum(dim=-1, keepdim=True)
        den = (pred * pred).sum(dim=-1, keepdim=True).clamp_min(1e-7)
        g = (num / den).clamp(0.3, 3.0)  # avoid crazy gains
        return pred * g

    # -------- Lightning hooks --------
    def on_fit_start(self):
        if isinstance(self.logger, WandbLogger):
            try:
                wb = self.logger.experiment
                wb.define_metric("trainer/global_step")
                wb.define_metric("*", step_metric="trainer/global_step")
            except Exception:
                pass

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        wav = self._sanitize_wave(batch["wav"])
        pss: List[str] = batch["pss"]

        pack = self._current_pack()
        preds = []
        for ps in pss:
            if not ps:
                continue
            ref = ref_slice_by_len(pack, ps)
            try:
                out = self.model(ps, ref, speed=1.0, return_output=True)
            except RuntimeError:
                continue
            pred = out.audio
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            preds.append(self._sanitize_wave(pred))

        if not preds:
            loss = torch.tensor(1.0, device=self.device, requires_grad=True)
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=wav.size(0))
            return loss

        preds_padded, wav_crop = self._pad_and_crop(preds, wav)
        if self.hparams.si_loss:
            if preds_padded.device != self._stft_dev:
                preds_padded = preds_padded.to(self._stft_dev)
            if wav_crop.device != self._stft_dev:
                wav_crop = wav_crop.to(self._stft_dev)
            preds_padded = self._align_gain(preds_padded, wav_crop)

        sc, mag = self._mrstft_loss(preds_padded, wav_crop, use_log10=self.hparams.loss_log10)
        l1 = F.l1_loss(preds_padded, wav_crop)
        loss = torch.nan_to_num(sc + mag + self.hparams.l1_weight * l1, nan=1e3, posinf=1e3, neginf=1e3)

        if self.hparams.mode == "mixture":
            w = self._current_weights()
            H = -torch.sum(w * torch.log(w + 1e-8))
            loss = loss + self.hparams.entropy_coeff * H

        bs = wav.size(0)
        self.log("trainer/global_step", float(self.global_step), on_step=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_stft_sc", sc, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_stft_mag", mag, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_l1", l1, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        wav = self._sanitize_wave(batch["wav"])
        pss: List[str] = batch["pss"]

        pack = self._current_pack()
        preds = []
        for ps in pss:
            if not ps:
                continue
            ref = ref_slice_by_len(pack, ps)
            try:
                out = self.model(ps, ref, speed=1.0, return_output=True)
            except RuntimeError:
                continue
            pred = out.audio
            if pred.dim() == 1:
                pred = pred.unsqueeze(0)
            preds.append(self._sanitize_wave(pred))

        if not preds:
            loss = torch.tensor(1.0, device=self.device)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=wav.size(0))
            return {"val_loss": loss}

        preds_padded, wav_crop = self._pad_and_crop(preds, wav)
        if self.hparams.si_loss:
            preds_padded = self._align_gain(preds_padded, wav_crop)

        sc, mag = self._mrstft_loss(preds_padded, wav_crop, use_log10=self.hparams.loss_log10)
        l1 = F.l1_loss(preds_padded, wav_crop)
        loss = torch.nan_to_num(sc + mag + self.hparams.l1_weight * l1, nan=1e3, posinf=1e3, neginf=1e3)

        bs = wav.size(0)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        self.log("val_stft_sc", sc, on_epoch=True, batch_size=bs)
        self.log("val_stft_mag", mag, on_epoch=True, batch_size=bs)
        self.log("val_l1", l1, on_epoch=True, batch_size=bs)
        return {"val_loss": l1}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        step = int(self.global_step)

        if isinstance(self.logger, WandbLogger) and self.hparams.mode == "mixture":
            w = self._current_weights().detach().cpu().tolist()
            metrics = {f"coef/{n}": float(v) for n, v in zip(self.voice_names, w)}
            metrics["trainer/global_step"] = step
            try:
                self.logger.log_metrics(metrics, step=step)
            except Exception:
                pass

        should_print = (step == 1) or (self.hparams.log_every and step % self.hparams.log_every == 0)
        if should_print and self.hparams.mode == "mixture":
            with torch.no_grad():
                w = self._current_weights().detach().cpu().numpy()
                pairs = list(zip(self.voice_names, w.tolist()))
                pairs.sort(key=lambda x: x[1], reverse=True)
                top = ", ".join([f"{n}:{v:.3f}" for n, v in pairs[:8]])
                print(f"[step {step}] weights: {top}")

        if self.hparams.mode == "mixture" and self.logits is not None:
            with torch.no_grad():
                self.logits.data = torch.nan_to_num(self.logits.data, nan=0.0, posinf=0.0, neginf=0.0)
                self.logits.data.clamp_(-20.0, 20.0)

        if torch.backends.mps.is_available() and (step % 25 == 0):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def on_after_backward(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)

    def configure_optimizers(self):
        groups = []
        if self.hparams.mode == "mixture":
            groups.append({"name": "logits", "params": [self.logits], "lr": self.hparams.logits_lr, "weight_decay": self.hparams.weight_decay})
        else:
            groups.append({"name": "emb", "params": [self.emb_param], "lr": self.hparams.emb_lr, "weight_decay": self.hparams.weight_decay})
        opt = torch.optim.AdamW(groups)

        total_steps = self.hparams.steps_per_epoch * self.hparams.max_epochs
        warmup_steps = max(1, int(self.hparams.warmup_frac * total_steps))

        if self.hparams.scheduler == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=max(g["lr"] for g in opt.param_groups),
                total_steps=total_steps, pct_start=self.hparams.warmup_frac,
                anneal_strategy="cos", div_factor=10.0, final_div_factor=25.0,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        elif self.hparams.scheduler == "cosine":
            base_lr = min(g["lr"] for g in opt.param_groups)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_steps, eta_min=base_lr * 0.05
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        elif self.hparams.scheduler == "cosine_warmup":
            def lr_lambda(s):
                if s < warmup_steps:
                    return float(s + 1) / float(warmup_steps)
                progress = (s - warmup_steps) / max(1, (total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        elif self.hparams.scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, verbose=False)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch", "monitor": "val_loss"}}

        else:
            return opt

    def on_train_start(self):
        # pick device for MR-STFT loss
        if self._stft_device_choice == "device":
            self._stft_dev = self.device
        elif self._stft_device_choice == "cpu":
            self._stft_dev = torch.device("cpu")
        else:
            self._stft_dev = self.device

        # move STFT windows once
        self.win_512  = self.win_512.to(self._stft_dev)
        self.win_1024 = self.win_1024.to(self._stft_dev)
        self.win_2048 = self.win_2048.to(self._stft_dev)

        # move base packs buffer to training device once
        if self.base_packs_buf is not None:
            self.base_packs_buf.data = self.base_packs_buf.data.to(self.device)

    def on_train_end(self):
        out_dir = Path(self.trainer.default_root_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.hparams.mode == "mixture":
            w = self._current_weights().detach().cpu()
            mixed = torch.einsum("n,ntcd->tcd", w, self.base_packs_buf.detach().cpu()).to(torch.float32)
            pt_path = out_dir / "optimized_mixture.pt"
            torch.save(mixed, str(pt_path))
            weights_json = out_dir / "optimized_mixture_weights.json"
            with open(weights_json, "w", encoding="utf-8") as f:
                json.dump({n: float(v) for n, v in zip(self.voice_names, w.tolist())}, f, indent=2)
            print(f"[done] saved mixed embedding -> {pt_path}")
            print(f"[done] saved weights -> {weights_json}")
        else:
            emb = self.emb_param.detach().cpu().to(torch.float32)
            pt_path = out_dir / "optimized_embedding.pt"
            torch.save(emb, str(pt_path))
            print(f"[done] saved optimized embedding -> {pt_path}")


# ---------------- Main / Trainer ----------------
def main():
    p = argparse.ArgumentParser(description="Lightning training to optimize Kokoro voice (mixture or full embedding).")
    # Dataset
    p.add_argument("--libritts_root", type=str,
                   default="/Users/mac/PycharmProjects/Jarvis_Phone/data/LibriTTS_R/dev-clean")
    p.add_argument("--speaker_id", type=str, default="251")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_train_items", type=int, default=0)  # ignored in quick test

    # Voices
    p.add_argument("--voices_dir", type=str,
                   default="/Users/mac/PycharmProjects/Jarvis_Phone/kokoro_us_voices",
                   help="Folder with *.pt voice packs (all will be used).")

    # Mode
    p.add_argument("--mode", choices=["mixture", "embedding"], default="mixture")

    # LRs
    p.add_argument("--logits_lr", type=float, default=2e-3)  # a bit higher for quick movement
    p.add_argument("--emb_lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Scheduler
    p.add_argument("--scheduler", type=str, default="onecycle",
                   choices=["onecycle", "cosine", "cosine_warmup", "plateau", "none"])
    p.add_argument("--warmup_frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=50)  # many epochs on tiny set

    # Softmax / exploration
    p.add_argument("--temp_start", type=float, default=1.5)
    p.add_argument("--temp_end", type=float, default=0.3)
    p.add_argument("--gumbel", type=str, default="false")

    # STFT device
    p.add_argument("--stft_device", type=str, default="auto",
                   choices=["auto", "device", "cpu"],
                   help="Where to compute MR-STFT loss. On MPS, 'cpu' is recommended.")

    # Loss
    p.add_argument("--si_loss", type=str, default="true")
    p.add_argument("--loss_log10", type=str, default="true")
    p.add_argument("--l1_weight", type=float, default=0.05)
    p.add_argument("--entropy_coeff", type=float, default=-0.001)

    # Logging
    p.add_argument("--log_wandb", type=str, default="true")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--demo_log_every", type=int, default=0)
    p.add_argument("--demo_text", type=str, default="Hi! This is an optimized Kokoro voice embedding.")

    # Val/checkpoints
    p.add_argument("--val_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=0)

    # Output
    p.add_argument("--out_dir", type=str, default="runs/opt_voice")

    args = p.parse_args()
    set_seed(1337)

    def s2b(x: str) -> bool:
        return str(x).lower() in {"1", "true", "yes", "y", "t"}

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    full = LTS_Spk_Ds(
        root_dir=Path(args.libritts_root),
        speaker_id=str(args.speaker_id),
        lang_code="a",
        voice="af_heart",
        quiet_pipeline=True,
        shuffle=True,
    )
    print(f"[dataset] {len(full)} items (speaker_id={getattr(full, 'speaker_id', 'NA')})")

    # Decide loss device
    if args.stft_device == "auto":
        use_cpu_stft = torch.backends.mps.is_available()
        args.stft_device = "cpu" if use_cpu_stft else "device"
    print(f"[stft] loss device = {args.stft_device}")

    # ---- QUICK TEST SAMPLING: pick the 50 SHORTEST items ----
    def _estimate_cost(i: int) -> int:
        """Cheap proxy for sample length. Prefer wav length, else transcript length, else 0."""
        try:
            items = getattr(full, "items", None)
            if isinstance(items, (list, tuple)):
                meta = items[i]
                if isinstance(meta, dict):
                    for k in ("wav_len", "wave_len", "num_samples", "n_samples", "samples"):
                        if k in meta and isinstance(meta[k], (int, float)):
                            return int(meta[k])
                    for k in ("text", "pss", "transcript", "utt"):
                        if k in meta and isinstance(meta[k], str):
                            return len(meta[k])
        except Exception:
            pass
        return 0

    all_indices = list(range(len(full)))
    # sort ascending by estimated length
    all_indices.sort(key=_estimate_cost)

    TRAIN_K = min(50, len(all_indices))
    VAL_K = min(8, max(0, len(all_indices) - TRAIN_K))

    train_idx = all_indices[:TRAIN_K]
    val_idx   = all_indices[TRAIN_K:TRAIN_K + VAL_K]
    print(f"[quick-test] training on {len(train_idx)} shortest samples; validation on {len(val_idx)}")

    use_pin = torch.cuda.is_available()  # pinning helps most on CUDA
    dl_train = DataLoader(
        Subset(full, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lts_collate_fn,
        pin_memory=use_pin,
        persistent_workers=(args.num_workers > 0),
    )
    dl_val = DataLoader(
        Subset(full, val_idx),
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        collate_fn=lts_collate_fn,
        pin_memory=use_pin,
        persistent_workers=(max(1, args.num_workers // 2) > 0),
    )

    steps_per_epoch = len(dl_train)
    base_packs, voice_names = load_base_packs(Path(args.voices_dir))

    module = KokoroEmbedOptimModule(
        mode=args.mode,
        base_packs=base_packs,
        voice_names=voice_names,
        steps_per_epoch=steps_per_epoch,
        max_epochs=args.epochs,

        # optimization
        logits_lr=args.logits_lr,
        emb_lr=args.emb_lr,
        weight_decay=args.weight_decay,

        # scheduler
        scheduler=args.scheduler,
        warmup_frac=args.warmup_frac,

        # softmax / exploration
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        gumbel=s2b(args.gumbel),
        entropy_coeff=args.entropy_coeff,

        # loss knobs
        si_loss=s2b(args.si_loss),
        loss_log10=s2b(args.loss_log10),
        l1_weight=args.l1_weight,
        stft_device=args.stft_device,

        # logging
        log_every=args.log_every,
        demo_log_every=args.demo_log_every,
        demo_text=args.demo_text,
    )

    # Logger
    log_wandb = s2b(args.log_wandb)
    if log_wandb:
        project = os.environ.get("WANDB_PROJECT", "kokoro_opt")
        logger = WandbLogger(project=project, save_dir=str(out_dir))
        logger.experiment.config.update({
            **vars(args),
            "sr": SR, "mode": args.mode, "n_voices": len(voice_names),
            "quick_test": True, "train_k": TRAIN_K, "val_k": VAL_K
        })
    else:
        logger = None

    # Callbacks
    callbacks = [LearningRateMonitor(logging_interval="step")]
    if args.save_every and args.save_every > 0:
        callbacks.append(ModelCheckpoint(
            dirpath=str(out_dir), filename="ckpt-step{step}", save_top_k=-1, every_n_train_steps=args.save_every
        ))
    callbacks.append(ModelCheckpoint(
        dirpath=str(out_dir), filename="best", monitor="val_loss", mode="min", save_top_k=1
    ))

    precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=max(1, min(args.val_every, steps_per_epoch)),  # validate periodically even on tiny set
        default_root_dir=str(out_dir),
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else 0.0,
        enable_progress_bar=True,
        log_every_n_steps=max(1, args.log_every),
        num_sanity_val_steps=0,
        precision=precision,
        benchmark=torch.cuda.is_available(),
        detect_anomaly=False,
    )

    print(f"[mode] {args.mode}")
    if args.mode == "mixture":
        print(f"[train] optimizing {len(voice_names)} mixture weights  (logits_lr={args.logits_lr})")
        print(f"[temp] start={args.temp_start} → end={args.temp_end}  gumbel={s2b(args.gumbel)}")
    else:
        print(f"[train] optimizing full embedding initialized from average of {len(voice_names)} voices (emb_lr={args.emb_lr})")
    print(f"[quick-test] steps/epoch={steps_per_epoch}, epochs={args.epochs}")

    trainer.fit(module, train_dataloaders=dl_train, val_dataloaders=dl_val)

    final_ckpt = out_dir / "final.ckpt"
    trainer.save_checkpoint(str(final_ckpt))
    print(f"[done] saved trainer checkpoint -> {final_ckpt}")


if __name__ == "__main__":
    main()