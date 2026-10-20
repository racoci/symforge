"""Multi-scale STFT feature extractor and contrastive perceptual embedding for audio signals."""

from __future__ import annotations
from typing import List, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from symforge.eval.loss import compute_infonce_loss


class MultiScaleSTFTExtractor(nn.Module):
    """Computes multi-resolution STFT log-magnitude representations and L2-normalized embeddings."""

    def __init__(
        self,
        fft_sizes: Sequence[int] = (512, 1024, 2048),
        hop_sizes: Optional[Sequence[int]] = None,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if not fft_sizes:
            raise ValueError("fft_sizes must be a non-empty sequence of integers")

        self.fft_sizes = tuple(int(s) for s in fft_sizes)
        if hop_sizes is None:
            self.hop_sizes = tuple(max(1, s // 4) for s in self.fft_sizes)
        else:
            if len(hop_sizes) != len(fft_sizes):
                raise ValueError(
                    f"hop_sizes length ({len(hop_sizes)}) must match fft_sizes length ({len(fft_sizes)})"
                )
            self.hop_sizes = tuple(int(h) for h in hop_sizes)

        self.eps = float(eps)

        # Register Hann windows as non-persistent buffers
        for n_fft in self.fft_sizes:
            self.register_buffer(f"window_{n_fft}", torch.hann_window(n_fft), persistent=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized multi-scale spectral embedding vector z = Phi(s) / ||Phi(s)||_2.

        Args:
            audio: 1D tensor of shape (T,) or 2D batch tensor of shape (B, T).

        Returns:
            Normalized latent embedding tensor of shape (B, D).
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim != 2:
            raise ValueError(f"Expected audio shape (T,) or (B, T), got {audio.shape}")

        features: List[torch.Tensor] = []

        for n_fft, hop in zip(self.fft_sizes, self.hop_sizes):
            win = getattr(self, f"window_{n_fft}")
            if win.device != audio.device:
                win = win.to(device=audio.device, dtype=audio.dtype)

            stft = torch.stft(
                audio,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=win,
                return_complex=True,
                pad_mode="reflect",
            )

            # Differentiable magnitude with epsilon inside sqrt to prevent zero-gradient NaN
            mag = torch.sqrt(stft.real**2 + stft.imag**2 + self.eps)

            # 1. Linear spectral energy profile across frequency bins
            lin_mean = mag.mean(dim=-1)
            lin_norm = F.normalize(lin_mean, p=2, dim=-1, eps=self.eps)

            # 2. Dynamic dB spectrum relative to peak energy (clamped to [-80, 0] dB range)
            ref = torch.clamp(mag.amax(dim=(-2, -1), keepdim=True), min=1e-6)
            db = 20.0 * torch.log10(torch.clamp(mag / ref, min=1e-4))
            db_norm = (db + 80.0) / 80.0
            db_mean = db_norm.mean(dim=-1)
            db_std = db_norm.std(dim=-1)

            features.append(lin_norm)
            features.append(db_mean)
            features.append(db_std)

        phi = torch.cat(features, dim=-1)
        return F.normalize(phi, p=2, dim=-1, eps=1e-8)

    def compute_loss(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Compute contrastive InfoNCE loss directly from raw audio signals.

        Args:
            query: Query audio signal of shape (T,) or (B, T).
            positive: Positive ground truth audio of shape (T,) or (B, T).
            negatives: Negative distractor audio signals of shape (K, T) or list/tuple of (T,).
            temperature: Softmax temperature parameter tau.

        Returns:
            Scalar tensor InfoNCE loss.
        """
        z_q = self.forward(query)
        z_pos = self.forward(positive)

        if isinstance(negatives, (list, tuple)):
            neg_embs = [self.forward(n) for n in negatives]
            z_negs = torch.cat(neg_embs, dim=0)
        elif negatives.ndim == 2:
            z_negs = self.forward(negatives)
        elif negatives.ndim == 3:
            b, k, t_len = negatives.shape
            reshaped = negatives.view(b * k, t_len)
            emb = self.forward(reshaped)
            z_negs = emb.view(b, k, -1)
        else:
            raise ValueError(f"Unsupported negatives shape: {negatives.shape}")

        return compute_infonce_loss(
            query=z_q,
            positive=z_pos,
            negatives=z_negs,
            temperature=temperature,
            normalize=True,
        )
