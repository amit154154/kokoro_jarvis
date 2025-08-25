#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyTorch Lightning SFT training for Kokoro KModel using your transcripts/audio.

- W&B logging ON by default (disable via --log_wandb false)
- Training plans: A (speaker/style), full, freeze_all
- Optional trainable voice pack embedding (--optimize_voice_pack)
- Periodic generation every K steps with W&B audio logging or disk saves
- Logs grad_norm and lr each step
"""
from datasets.KokoroJarvisSFTDataset import KokoroJarvisSFTDataset,collate_fn
from utills import *
from losses.MultiResSTFTLoss import MultiResSTFTLoss

import os, argparse, random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import torch.nn.functional as F

import soundfile as sf
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from kokoro import KPipeline
from kokoro.model import KModel

SR = 24000

# ---------------- Lightning Module ---------------------

class KokoroSFTModule(pl.LightningModule):
    def __init__(self, lr: float = 2e-4, grad_clip: float = 1.0, stft_on_mps: bool = False,
                 export_val_samples: bool = False, out_dir: Path = Path("runs/kokoro_sft"),
                 voice: str = "af_heart", lang_code: str = "a",
                 gen_every: int = 0, gen_text: Optional[List[str]] = None,
                 train_plan: str = "A", optimize_voice: bool = False,
                 log_pairs_every: int = 0, log_pairs_n: int = 2):
        super().__init__()
        self.save_hyperparameters(ignore=["out_dir"])  # log friendly

        # Model
        self.model = KModel(repo_id='hexgrad/Kokoro-82M')

        # Loss
        device_for_loss = self.device if (stft_on_mps and torch.backends.mps.is_available()) else None
        self.stft_loss = MultiResSTFTLoss(device=device_for_loss)

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.export_val_samples = export_val_samples
        self.voice = voice

        self.train_plan = (train_plan or "A").lower()

        self.lang_code = lang_code
        self.gen_every = int(gen_every) if gen_every is not None else 0
        self.gen_text = list(gen_text) if gen_text else []

        # Quiet pipeline for G2P during periodic generation (uses this model for synth)
        self.gen_pipeline = KPipeline(lang_code=self.lang_code, model=False)
        self.gen_dir = (self.out_dir / "gen"); self.gen_dir.mkdir(parents=True, exist_ok=True)

        # Optional trainable voice pack embedding
        self.optimize_voice = bool(optimize_voice)
        self.voice_pack_param: Optional[torch.nn.Parameter] = None
        if self.optimize_voice:
            init_pack = self.gen_pipeline.load_voice(self.voice)  # CPU FloatTensor
            self.voice_pack_param = torch.nn.Parameter(init_pack.clone().detach())

        # Pair logging settings
        self.log_pairs_every = int(log_pairs_every or 0)
        self.log_pairs_n = int(log_pairs_n or 2)
        self.pairs_dir = (self.out_dir / "pairs"); self.pairs_dir.mkdir(parents=True, exist_ok=True)

        # Apply training plan (freeze/unfreeze)
        self._apply_freeze_plan()

        self._val_export_count = 0

    # ---------- helpers ----------

    def _apply_freeze_plan(self):
        """Apply parameter freezing according to train_plan.
        Plans:
          - 'a': Speaker/style adaptation (bert_encoder + predictor only)
          - 'full': Full finetune (all trainable)
          - 'freeze_all': Freeze entire model (useful for testing generation/logging)
        """
        plan = self.train_plan
        M = self.model
        # Default: freeze everything
        for p in M.parameters():
            p.requires_grad = False
        if plan == "a":
            for p in M.bert_encoder.parameters(): p.requires_grad = True
            for p in M.predictor.parameters():    p.requires_grad = True
        elif plan == "full":
            for p in M.parameters(): p.requires_grad = True
        elif plan == "freeze_all":
            pass
        else:
            # Fallback to plan A
            for p in M.bert_encoder.parameters(): p.requires_grad = True
            for p in M.predictor.parameters():    p.requires_grad = True

        # Keep trainable voice pack if requested
        if self.voice_pack_param is not None:
            self.voice_pack_param.requires_grad = True

        trainable = sum(p.numel() for p in M.parameters() if p.requires_grad)
        total = sum(p.numel() for p in M.parameters())
        print(f"[plan={plan}] trainable params: {trainable:,} / {total:,}")

    @staticmethod
    def ref_slice_by_len(pack: torch.FloatTensor, ps: str) -> torch.FloatTensor:
        ref = pack[len(ps) - 1]
        return ref.unsqueeze(0) if ref.dim() == 1 else ref

    def _forward_batch(self, pss: List[str], pack: torch.Tensor) -> torch.Tensor:
        preds = []
        for ps in pss:
            ref = self.ref_slice_by_len(pack, ps).to(self.device)
            out = self.model(ps, ref, speed=1.0, return_output=True)
            pred = out.audio
            if pred.dim() == 1: pred = pred.unsqueeze(0)
            preds.append(pred)
        max_len = max(x.shape[-1] for x in preds)
        preds_padded = torch.stack([F.pad(x, (0, max_len - x.shape[-1])) for x in preds], dim=0).squeeze(1)
        return preds_padded

    def _periodic_generate(self):
        """Generate predefined text and either log to W&B or save WAVs to disk."""
        if not self.gen_text: return
        step = int(self.global_step)
        out_step_dir = self.gen_dir / f"step_{step:06d}"
        out_step_dir.mkdir(parents=True, exist_ok=True)

        is_wandb = isinstance(self.logger, WandbLogger)
        wb_list = []

        for i, txt in enumerate(self.gen_text):
            try:
                # Ensure voice arg type is correct for KPipeline
                if self.voice_pack_param is not None:
                    voice_arg = self.voice_pack_param.detach().cpu().to(torch.float32)
                    if not isinstance(voice_arg, torch.FloatTensor):
                        voice_arg = voice_arg.type(torch.FloatTensor)
                else:
                    voice_arg = self.voice

                j = 0
                for res in self.gen_pipeline(text=txt, voice=voice_arg, model=self.model):
                    if res.audio is None: continue
                    wav = res.audio.detach().float().cpu().numpy()
                    out_path = out_step_dir / f"sample{i}_chunk{j}.wav"
                    sf.write(str(out_path), wav, SR)
                    if is_wandb:
                        try:
                            import wandb
                            wb_list.append((f"gen/sample{i}_chunk{j}", wandb.Audio(wav, sample_rate=SR)))
                            wb_list.append((f"gen/sample{i}_chunk{j}_text", txt))
                        except Exception:
                            pass
                    j += 1
            except Exception as e:
                print(f"[gen@{step}] error for text {i}: {e}")

        if is_wandb and wb_list:
            # merge to dict; text entries included
            log_dict = {name: val for name, val in wb_list}
            log_dict["gen_step"] = step
            self.logger.experiment.log(log_dict, step=step)

    def _log_train_pairs(self, wav: torch.Tensor, pred: torch.Tensor, texts: List[str]):
        """Log GT vs Pred pairs + text every self.log_pairs_every steps."""
        if self.log_pairs_every <= 0: return
        step = int(self.global_step)
        if step == 0 or (step % self.log_pairs_every) != 0: return

        B = wav.size(0)
        n = min(self.log_pairs_n, B)
        out_step = self.pairs_dir / f"step_{step:06d}"
        out_step.mkdir(parents=True, exist_ok=True)

        is_wandb = isinstance(self.logger, WandbLogger)
        wb_entries = []
        for i in range(n):
            gt = wav[i].detach().float().cpu().numpy()
            pr = pred[i].detach().float().cpu().numpy()
            text = texts[i] if i < len(texts) else ""
            # always save to disk
            gt_p = out_step / f"pair{i}_gt.wav"
            pr_p = out_step / f"pair{i}_pred.wav"
            txt_p = out_step / f"pair{i}.txt"
            sf.write(str(gt_p), gt, SR)
            sf.write(str(pr_p), pr, SR)
            with open(txt_p, "w", encoding="utf-8") as f:
                f.write(text)

            if is_wandb:
                try:
                    import wandb
                    wb_entries.append((f"pairs/pair{i}_gt", wandb.Audio(gt, sample_rate=SR)))
                    wb_entries.append((f"pairs/pair{i}_pred", wandb.Audio(pr, sample_rate=SR)))
                    wb_entries.append((f"pairs/pair{i}_text", text))
                except Exception:
                    pass

        if is_wandb and wb_entries:
            log_dict = {k: v for k, v in wb_entries}
            log_dict["pairs_step"] = step
            self.logger.experiment.log(log_dict, step=step)

    # ---------- Lightning required ----------
    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        wav = batch["wav"].to(self.device)  # [B, T]
        pss: List[str] = batch["pss"]
        texts: List[str] = batch["texts"]

        if self.voice_pack_param is not None:
            pack = self.voice_pack_param.to(self.device)
        else:
            pack = batch["voice_pack"].to(self.device)

        preds_padded = self._forward_batch(pss, pack)
        # match lengths
        if wav.shape[-1] < preds_padded.shape[-1]:
            wav = F.pad(wav, (0, preds_padded.shape[-1] - wav.shape[-1]))
        elif wav.shape[-1] > preds_padded.shape[-1]:
            preds_padded = F.pad(preds_padded, (0, wav.shape[-1] - preds_padded.shape[-1]))

        loss_stft, stft_metrics = self.stft_loss(preds_padded, wav)
        loss_l1 = F.l1_loss(preds_padded, wav)
        loss = loss_stft + 0.1 * loss_l1

        bs = wav.size(0)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_stft_sc", stft_metrics["stft_sc"], on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_stft_mag", stft_metrics["stft_mag"], on_step=True, on_epoch=True, batch_size=bs)
        self.log("train_l1", loss_l1, on_step=True, on_epoch=True, batch_size=bs)

        # Paired logging
        self._log_train_pairs(wav, preds_padded, texts)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        wav = batch["wav"].to(self.device)
        pss: List[str] = batch["pss"]

        if self.voice_pack_param is not None:
            pack = self.voice_pack_param.to(self.device)
        else:
            pack = batch["voice_pack"].to(self.device)

        preds_padded = self._forward_batch(pss, pack)
        if wav.shape[-1] < preds_padded.shape[-1]:
            wav = F.pad(wav, (0, preds_padded.shape[-1] - wav.shape[-1]))
        elif wav.shape[-1] > preds_padded.shape[-1]:
            preds_padded = F.pad(preds_padded, (0, wav.shape[-1] - preds_padded.shape[-1]))

        loss_stft, _ = self.stft_loss(preds_padded, wav)
        loss_l1 = F.l1_loss(preds_padded, wav)
        loss = loss_stft + 0.1 * loss_l1

        bs = wav.size(0)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False, batch_size=bs)
        return {"val_loss": loss}

    def on_validation_epoch_start(self) -> None:
        self._val_export_count = 0

    def on_after_backward(self):
        # global L2 grad norm over trainable params
        total_sq = 0.0
        for p in self.parameters():
            if p.requires_grad and p.grad is not None:
                g = p.grad.detach().float()
                total_sq += float(torch.sum(g * g).item())
        if total_sq > 0.0:
            self.log("grad_norm", total_sq ** 0.5, on_step=True, on_epoch=False, prog_bar=False)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.gen_every and self.global_step > 0 and (self.global_step % self.gen_every) == 0:
            self._periodic_generate()
        # log learning rate
        try:
            if self.trainer is not None and self.trainer.optimizers:
                lr = self.trainer.optimizers[0].param_groups[0]["lr"]
                self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=False)
        except Exception:
            pass

    def configure_optimizers(self):
        # voice pack gets smaller LR to prevent rapid drift
        main_params = [p for n, p in self.named_parameters() if p.requires_grad and n != "voice_pack_param"]
        groups = [{"params": main_params, "lr": self.hparams.lr, "weight_decay": 1e-4}]
        if self.voice_pack_param is not None and self.voice_pack_param.requires_grad:
            groups.append({"params": [self.voice_pack_param], "lr": self.hparams.lr * 0.25, "weight_decay": 0.0})
        return torch.optim.AdamW(groups)

# ---------------- Main / Trainer -----------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_root", default="data/JarvisSounds", type=str)
    p.add_argument("--metadata_csv", default="data/process_JarvisSounds/metadata.csv", type=str)
    p.add_argument("--out_dir", default="runs/kokoro_sft", type=str)
    p.add_argument("--voice", default="af_heart", type=str)
    p.add_argument("--lang_code", default="a", type=str)  # American English
    p.add_argument("--epochs", default=10, type=int)
    p.add_argument("--batch_size", default=2, type=int)
    p.add_argument("--lr", default=2e-4, type=float)
    p.add_argument("--grad_clip", default=1.0, type=float)
    p.add_argument("--save_every", default=1000, type=int)
    p.add_argument("--val_every", default=150, type=int, help="validate every N training steps")
    p.add_argument("--num_workers", default=0, type=int)
    p.add_argument("--max_train_items", default=0, type=int, help="debug cap; 0=all")
    p.add_argument("--export_val_samples", action="store_true")
    p.add_argument(
        "--log_wandb",
        nargs="?",
        const=True,
        default=True,
        type=str2bool_default_true,
        help="Use Weights & Biases logging (default: True). Pass '--log_wandb false' to disable."
    )
    p.add_argument("--gen_every", default=10, type=int, help="Every K steps, generate predefined text (0=off)")
    p.add_argument("--gen_text", nargs='*', default=["hello my name is Jarvis"], type=str, help="Texts to synthesize at each generation interval")
    p.add_argument("--train_plan", default="A", choices=["A", "full", "freeze_all"], help="Freeze plan: A (speaker/style), full finetune, or freeze_all")
    p.add_argument(
        "--optimize_voice_pack",
        nargs="?",
        const=True,
        default=False,
        type=str2bool_default_false,
        help="If set, make the selected voice_pack a trainable parameter initialized from the current voice."
    )
    p.add_argument("--log_pairs_every", default=5, type=int,
                   help="Every K steps log GT vs Pred pairs from training batch (0=off)")
    p.add_argument("--log_pairs_n", default=1, type=int,
                   help="Max number of pairs to log per logging step")
    args = p.parse_args()

    set_seed(1337)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset / splits
    full = KokoroJarvisSFTDataset(
        audio_root=Path(args.audio_root),
        metadata_csv=Path(args.metadata_csv),
        lang_code=args.lang_code,
        voice=args.voice,
        quiet_pipeline=True
    )
    indices = list(range(len(full)))
    random.shuffle(indices)
    n_val = max(1, int(0.02 * len(indices)))
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    if args.max_train_items > 0:
        train_idx = train_idx[:args.max_train_items]

    subset_train = torch.utils.data.Subset(full, train_idx)
    subset_val = torch.utils.data.Subset(full, val_idx)

    dl_train = DataLoader(subset_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=False,
                          persistent_workers=(args.num_workers > 0))
    dl_val = DataLoader(subset_val, batch_size=1, shuffle=False,
                        num_workers=max(1, args.num_workers // 2), collate_fn=collate_fn, pin_memory=False,
                        persistent_workers=(max(1, args.num_workers // 2) > 0))

    # Lightning module
    module = KokoroSFTModule(
        lr=args.lr,
        grad_clip=args.grad_clip,
        stft_on_mps=torch.backends.mps.is_available(),
        export_val_samples=args.export_val_samples,
        out_dir=out_dir,
        voice=args.voice,
        lang_code=args.lang_code,
        gen_every=args.gen_every,
        gen_text=args.gen_text,
        train_plan=args.train_plan,
        optimize_voice=args.optimize_voice_pack,
        log_pairs_every=args.log_pairs_every,
        log_pairs_n=args.log_pairs_n,
    )

    # Logger(s)
    logger = None
    if args.log_wandb:
        project = os.environ.get("WANDB_PROJECT", "kokoro_sft")
        logger = WandbLogger(project=project, save_dir=str(out_dir))
        logger.experiment.config.update({
            **vars(args),
            "sr": SR,
        })

    # Callbacks
    ckpt_best = ModelCheckpoint(
        dirpath=str(out_dir), filename="kokoro_sft-best", monitor="val_loss", mode="min", save_top_k=1
    )
    ckpt_steps = ModelCheckpoint(
        dirpath=str(out_dir), filename="kokoro_sft-step{step}", save_top_k=-1, every_n_train_steps=args.save_every
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[ckpt_best, ckpt_steps, lr_monitor],
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else 0.0,
        val_check_interval=args.val_every,
        default_root_dir=str(out_dir),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    print(f"Using training plan: {args.train_plan}")
    print(f"Optimize voice pack: {'ON' if args.optimize_voice_pack else 'OFF'}")
    print(f"Pairs logging every: {args.log_pairs_every} steps (N={args.log_pairs_n})")
    print(f"W&B logging: {'ON' if args.log_wandb else 'OFF'}")

    trainer.fit(module, train_dataloaders=dl_train, val_dataloaders=dl_val)

    # Save final checkpoint
    final_ckpt = out_dir / "kokoro_sft-final.ckpt"
    trainer.save_checkpoint(str(final_ckpt))
    print(f"[done] saved {final_ckpt}")


if __name__ == "__main__":
    main()