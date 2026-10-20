"""Tests for 1D DSP audio primitives."""

import math
import pytest
import torch

from symforge.core.ast import FreeParam
from symforge.audio.primitives import (
    SineOscillator,
    SawOscillator,
    SquareOscillator,
    NoiseGenerator,
    ADSREnvelope,
    ExpDecayEnvelope,
    LFO,
    FMOperator,
    FMSynthesisPair,
    LowPassFilter,
    GainModule,
    MixModule,
)


def _estimate_frequency_fft(x: torch.Tensor, fs: float) -> float:
    """Estimate fundamental frequency of 1D signal using padded FFT peak."""
    x_centered = x - x.mean()
    n = len(x_centered)
    pad_factor = 64
    padded_n = pad_factor * n
    x_pad = torch.fft.rfft(x_centered, n=padded_n)
    freqs = torch.fft.rfftfreq(padded_n, d=1.0 / fs)
    peak_idx = torch.argmax(torch.abs(x_pad))
    return float(freqs[peak_idx].item())


class TestOscillators:
    """Test suite for oscillator primitives."""

    @pytest.mark.parametrize("f_target", [110.0, 220.0, 440.0, 880.0])
    def test_sine_oscillator_frequency_and_bounds(self, f_target: float):
        fs = 16000.0
        duration = 1.0
        t = torch.arange(int(fs * duration)) / fs

        osc = SineOscillator(freq=f_target, phase=0.0)
        signal = osc(t)

        # Check amplitude bounds strictly within [-1.0, 1.0]
        assert signal.min().item() >= -1.0 - 1e-6
        assert signal.max().item() <= 1.0 + 1e-6

        # Check frequency precision within 0.1 Hz
        f_est = _estimate_frequency_fft(signal, fs)
        assert abs(f_est - f_target) < 0.1

    @pytest.mark.parametrize("f_target", [110.0, 220.0, 440.0])
    def test_saw_oscillator_frequency_and_bounds(self, f_target: float):
        fs = 16000.0
        duration = 1.0
        t = torch.arange(int(fs * duration)) / fs

        osc = SawOscillator(freq=f_target, phase=0.0)
        signal = osc(t)

        assert signal.min().item() >= -1.0 - 1e-6
        assert signal.max().item() <= 1.0 + 1e-6

        f_est = _estimate_frequency_fft(signal, fs)
        assert abs(f_est - f_target) < 0.1

    @pytest.mark.parametrize("duty", [0.25, 0.5, 0.75])
    def test_square_oscillator_frequency_and_duty(self, duty: float):
        fs = 16000.0
        duration = 1.0
        t = torch.arange(int(fs * duration)) / fs
        f_target = 440.0

        osc = SquareOscillator(freq=f_target, duty=duty)
        signal = osc(t)

        assert signal.min().item() >= -1.0 - 1e-6
        assert signal.max().item() <= 1.0 + 1e-6

        f_est = _estimate_frequency_fft(signal, fs)
        assert abs(f_est - f_target) < 0.1

        # Check positive cycle fraction
        pos_fraction = (signal > 0.0).float().mean().item()
        assert abs(pos_fraction - duty) < 0.05

    def test_noise_generator_bounds_and_determinism(self):
        t = torch.linspace(0, 0.1, 1600)
        gen1 = NoiseGenerator(seed=123)
        gen2 = NoiseGenerator(seed=123)
        gen3 = NoiseGenerator(seed=456)

        n1 = gen1(t)
        n2 = gen2(t)
        n3 = gen3(t)

        assert n1.min().item() >= -1.0
        assert n1.max().item() <= 1.0
        assert torch.allclose(n1, n2)
        assert not torch.allclose(n1, n3)

    def test_oscillator_autograd_gradients(self):
        t = torch.linspace(0, 0.05, 800)

        sine = SineOscillator(freq=440.0, phase=0.1)
        saw = SawOscillator(freq=440.0, phase=0.1)
        sq = SquareOscillator(freq=440.0, duty=0.5, phase=0.1)

        loss_sine = (sine(t) ** 2).sum()
        loss_sine.backward()
        assert sine.freq.grad is not None and sine.freq.grad.abs().item() > 0.0
        assert sine.phase.grad is not None and sine.phase.grad.abs().item() > 0.0

        loss_saw = (saw(t) ** 2).sum()
        loss_saw.backward()
        assert saw.freq.grad is not None and saw.freq.grad.abs().item() > 0.0
        assert saw.phase.grad is not None and saw.phase.grad.abs().item() > 0.0

        loss_sq = (sq(t) ** 2).sum()
        loss_sq.backward()
        assert sq.freq.grad is not None and sq.freq.grad.abs().item() > 0.0
        assert sq.duty.grad is not None and sq.duty.grad.abs().item() > 0.0


class TestEnvelopes:
    """Test suite for envelope primitives."""

    def test_adsr_envelope_stages_and_bounds(self):
        t = torch.linspace(0, 1.0, 1000)
        att, dec, sus, rel = 0.1, 0.1, 0.6, 0.2
        gate = 0.5
        adsr = ADSREnvelope(attack=att, decay=dec, sustain=sus, release=rel)
        env = adsr(t, gate_duration=gate)

        # Bounds: strictly within [0.0, 1.0]
        assert env.min().item() >= 0.0 - 1e-6
        assert env.max().item() <= 1.0 + 1e-6

        # Peak at attack end: t = 0.1
        idx_peak = int(0.1 * 1000)
        assert abs(env[idx_peak].item() - 1.0) < 0.05

        # Sustain level during sustain phase: t = 0.3
        idx_sus = int(0.3 * 1000)
        assert abs(env[idx_sus].item() - sus) < 0.05

        # Final release state: t = 0.9 (> gate + rel = 0.7)
        idx_end = int(0.9 * 1000)
        assert env[idx_end].item() < 1e-3

    def test_adsr_autograd_gradients(self):
        t = torch.linspace(0, 1.0, 1000)
        adsr = ADSREnvelope(attack=0.1, decay=0.1, sustain=0.5, release=0.2)
        env = adsr(t, gate_duration=0.5)
        loss = env.sum()
        loss.backward()

        assert adsr.attack.grad is not None and adsr.attack.grad.abs().item() > 0.0
        assert adsr.decay.grad is not None and adsr.decay.grad.abs().item() > 0.0
        assert adsr.sustain.grad is not None and adsr.sustain.grad.abs().item() > 0.0
        assert adsr.release.grad is not None and adsr.release.grad.abs().item() > 0.0

    def test_exp_decay_envelope(self):
        t = torch.linspace(0, 1.0, 1000)
        exp_env = ExpDecayEnvelope(decay_rate=5.0)
        env = exp_env(t)

        assert env.min().item() >= 0.0
        assert env.max().item() <= 1.0
        assert env[0].item() == pytest.approx(1.0, abs=1e-4)
        assert env[-1].item() == pytest.approx(math.exp(-5.0), abs=1e-3)

        loss = env.sum()
        loss.backward()
        assert exp_env.decay_rate.grad is not None and exp_env.decay_rate.grad.abs().item() > 0.0


class TestModulatorsAndFilters:
    """Test suite for modulators, filters, and mixers."""

    def test_lfo_modulation(self):
        t = torch.linspace(0, 1.0, 16000)
        lfo = LFO(freq=5.0, depth=0.8, waveform="sine")
        out = lfo(t)

        assert out.min().item() >= -0.8 - 1e-4
        assert out.max().item() <= 0.8 + 1e-4

        f_est = _estimate_frequency_fft(out, fs=16000.0)
        assert abs(f_est - 5.0) < 0.1

    def test_fm_operator_bounds_and_gradients(self):
        t = torch.linspace(0, 0.1, 1600)
        fm = FMOperator(carrier_freq=440.0, mod_freq=220.0, mod_depth=2.0)
        signal = fm(t)

        assert signal.min().item() >= -1.0 - 1e-6
        assert signal.max().item() <= 1.0 + 1e-6

        loss = (signal ** 2).sum()
        loss.backward()
        assert fm.carrier_freq.grad is not None and fm.carrier_freq.grad.abs().item() > 0.0
        assert fm.mod_freq.grad is not None and fm.mod_freq.grad.abs().item() > 0.0
        assert fm.mod_depth.grad is not None and fm.mod_depth.grad.abs().item() > 0.0

        # Check alias FMSynthesisPair works
        assert issubclass(FMSynthesisPair, FMOperator)

    def test_lowpass_filter_attenuation(self):
        fs = 16000.0
        duration = 1.0
        t = torch.arange(int(fs * duration)) / fs

        f_low = 200.0
        f_high = 4000.0
        sig = torch.sin(2 * math.pi * f_low * t) + torch.sin(2 * math.pi * f_high * t)

        lpf = LowPassFilter(cutoff=800.0, resonance=1.0, sample_rate=fs)
        filtered = lpf(sig, t=t)

        # Spectral analysis of filtered output
        fft_out = torch.abs(torch.fft.rfft(filtered))
        low_mag = fft_out[int(f_low * duration)].item()
        high_mag = fft_out[int(f_high * duration)].item()

        # Attenuation ratio should be at least 10x (> 20 dB suppression of 4000Hz vs 200Hz)
        assert low_mag > 10.0 * high_mag

    def test_lowpass_filter_autograd_gradients(self):
        fs = 16000.0
        t = torch.arange(16000) / fs
        sig = torch.sin(2 * math.pi * 500.0 * t) + torch.sin(2 * math.pi * 3000.0 * t)

        lpf = LowPassFilter(cutoff=1000.0, resonance=1.0, sample_rate=fs)
        out = lpf(sig, t=t)
        loss = (out ** 2).sum()
        loss.backward()

        assert lpf.cutoff.grad is not None and lpf.cutoff.grad.abs().item() > 0.0
        assert lpf.resonance.grad is not None and lpf.resonance.grad.abs().item() > 0.0

    def test_gain_and_mix_modules(self):
        t = torch.linspace(0, 0.1, 1600)
        osc1 = SineOscillator(freq=440.0)
        osc2 = SquareOscillator(freq=220.0)
        s1 = osc1(t)
        s2 = osc2(t)

        gain = GainModule(gain=0.5)
        s1_gained = gain(s1)
        assert s1_gained.max().item() <= 0.5 + 1e-4

        mix = MixModule(ratio=0.7)
        s_mixed = mix(s1, s2)
        assert s_mixed.min().item() >= -1.0 - 1e-6
        assert s_mixed.max().item() <= 1.0 + 1e-6

        loss = s_mixed.sum()
        loss.backward()
        assert mix.ratio.grad is not None and mix.ratio.grad.abs().item() > 0.0


class TestBatchAndFreeParamIntegration:
    """Test batch evaluation and FreeParam integration."""

    def test_batch_evaluation_shape(self):
        batch_size = 4
        num_samples = 8000
        t = torch.linspace(0, 0.5, num_samples).unsqueeze(0).expand(batch_size, -1)

        osc = SineOscillator(freq=440.0)
        out = osc(t)
        assert out.shape == (batch_size, num_samples)

        lpf = LowPassFilter(cutoff=1000.0, sample_rate=16000.0)
        filtered = lpf(out, t=t)
        assert filtered.shape == (batch_size, num_samples)

    def test_freeparam_initialization(self):
        p_freq = FreeParam("f_sine", 20.0, 2000.0, 440.0)
        p_phase = FreeParam("phi_sine", 0.0, 6.28, 1.57)

        osc = SineOscillator(freq=p_freq, phase=p_phase)
        assert osc.freq.item() == pytest.approx(440.0)
        assert osc.phase.item() == pytest.approx(1.57)
        assert hasattr(osc.freq, "_free_param")
        assert osc.freq._free_param.id == "f_sine"
