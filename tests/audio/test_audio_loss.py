"""Tests for Multi-scale STFT feature extraction and contrastive audio loss."""

import math
import pytest
import torch

from symforge.audio.primitives import (
    SineOscillator,
    SawOscillator,
    SquareOscillator,
    LowPassFilter,
)
from symforge.audio.stft import MultiScaleSTFTExtractor
from symforge.eval.loss import compute_infonce_loss, ContrastivePerceptualLoss


class TestMultiScaleSTFTExtractor:
    """Test suite for Multi-scale STFT feature extraction."""

    def test_extractor_output_shape_and_normalization(self):
        fs = 16000
        duration = 0.5
        t = torch.arange(int(fs * duration)) / float(fs)
        osc = SineOscillator(freq=440.0)
        audio = osc(t)

        extractor = MultiScaleSTFTExtractor(fft_sizes=(512, 1024, 2048))
        emb = extractor(audio)

        # Output shape: (1, D) for 1D audio input
        assert emb.ndim == 2
        assert emb.shape[0] == 1
        # L2-normalization invariant: ||z||_2 == 1.0
        norm = torch.norm(emb, p=2, dim=-1).item()
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_batch_extraction(self):
        fs = 16000
        duration = 0.5
        b = 3
        t = torch.arange(int(fs * duration)) / float(fs)
        t_batch = t.unsqueeze(0).expand(b, -1)

        osc = SineOscillator(freq=440.0)
        audio_batch = osc(t_batch)

        extractor = MultiScaleSTFTExtractor(fft_sizes=(512, 1024))
        emb_batch = extractor(audio_batch)

        assert emb_batch.shape[0] == b
        norms = torch.norm(emb_batch, p=2, dim=-1)
        for n in norms:
            assert n.item() == pytest.approx(1.0, abs=1e-5)

    def test_contrastive_separation_saw_vs_square(self):
        """Invariant: Differentiates between Sawtooth and Square waves of equal pitch with contrastive separation margin > 2.0 under InfoNCE."""
        fs = 16000.0
        duration = 1.0
        t = torch.arange(int(fs * duration)) / fs
        pitch = 440.0

        saw_osc = SawOscillator(freq=pitch)
        sq_osc = SquareOscillator(freq=pitch, duty=0.5)
        sine_osc = SineOscillator(freq=pitch)

        s_saw = saw_osc(t)
        s_sq = sq_osc(t)
        s_sine = sine_osc(t)

        extractor = MultiScaleSTFTExtractor(fft_sizes=(512, 1024, 2048))
        z_saw = extractor(s_saw)
        z_sq = extractor(s_sq)
        z_sine = extractor(s_sine)

        tau = 0.07
        sim_pos = torch.sum(z_saw * z_saw, dim=-1).item()
        sim_neg_sq = torch.sum(z_saw * z_sq, dim=-1).item()
        sim_neg_sine = torch.sum(z_saw * z_sine, dim=-1).item()

        # Logit margin under InfoNCE: (sim_pos - sim_neg) / tau
        margin_sq = (sim_pos - sim_neg_sq) / tau
        margin_sine = (sim_pos - sim_neg_sine) / tau

        assert margin_sq > 2.0, f"Expected margin > 2.0, got {margin_sq}"
        assert margin_sine > 2.0, f"Expected margin > 2.0, got {margin_sine}"

        # Evaluate InfoNCE loss with Saw as positive and Square as negative
        negatives = torch.cat([z_sq, z_sine], dim=0)
        loss = compute_infonce_loss(
            query=z_saw,
            positive=z_saw,
            negatives=negatives,
            temperature=tau,
        )
        # Loss should be near 0 for correct target
        assert loss.item() < 0.01

        # Mismatched target: Square wave trying to match Saw target when competing against true Saw
        loss_mismatch = compute_infonce_loss(
            query=z_sq,
            positive=z_saw,
            negatives=z_sq,
            temperature=tau,
        )
        assert (loss_mismatch.item() - loss.item()) > 2.0

    def test_autograd_flow_through_stft_loss(self):
        """Verify non-zero gradients backpropagate through STFT extractor into audio oscillator parameters."""
        fs = 16000.0
        duration = 0.5
        t = torch.arange(int(fs * duration)) / fs

        # Synthesize target with known parameters
        with torch.no_grad():
            target_osc = SawOscillator(freq=440.0)
            target_audio = target_osc(t)

        # Candidate oscillator with perturbed parameter
        cand_osc = SawOscillator(freq=430.0)

        extractor = MultiScaleSTFTExtractor(fft_sizes=(512, 1024))
        z_cand = extractor(cand_osc(t))

        with torch.no_grad():
            z_target = extractor(target_audio)
            distractor_audio = SineOscillator(freq=220.0)(t)
            z_distractor = extractor(distractor_audio)

        # Compute loss
        loss = compute_infonce_loss(
            query=z_cand,
            positive=z_target,
            negatives=z_distractor,
            temperature=0.07,
        )

        loss.backward()
        assert cand_osc.freq.grad is not None
        assert cand_osc.freq.grad.abs().item() > 0.0

    def test_filter_autograd_through_stft(self):
        """Verify gradients flow through filter cutoff into STFT embedding."""
        fs = 16000.0
        duration = 0.5
        t = torch.arange(int(fs * duration)) / fs
        raw_sig = SawOscillator(freq=300.0)(t)

        lpf = LowPassFilter(cutoff=1200.0, sample_rate=fs)
        filtered = lpf(raw_sig, t=t)

        extractor = MultiScaleSTFTExtractor(fft_sizes=(512, 1024))
        emb = extractor(filtered)

        target_emb = extractor(raw_sig).detach()
        neg_emb = extractor(SineOscillator(freq=100.0)(t)).detach()

        loss = compute_infonce_loss(emb, target_emb, neg_emb)
        loss.backward()

        assert lpf.cutoff.grad is not None
        assert lpf.cutoff.grad.abs().item() > 0.0

    def test_contrastive_perceptual_loss_module_integration(self):
        fs = 16000.0
        duration = 0.5
        t = torch.arange(int(fs * duration)) / fs

        extractor = MultiScaleSTFTExtractor(fft_sizes=(512, 1024))
        cpl = ContrastivePerceptualLoss(temperature=0.07)

        s1 = SineOscillator(freq=440.0)(t)
        s2 = SineOscillator(freq=440.0)(t)
        s_neg = SineOscillator(freq=880.0)(t)

        z1 = extractor(s1)
        z2 = extractor(s2)
        z_neg = extractor(s_neg)

        loss = cpl(z1, z2, z_neg)
        assert loss.item() < 0.01
