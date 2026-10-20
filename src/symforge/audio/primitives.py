"""Functional and differentiable PyTorch 1D DSP primitives for audio synthesis."""

from __future__ import annotations
import math
from typing import Any, Optional, Union
import torch
import torch.nn as nn

from symforge.core.ast import FreeParam


def _ensure_param(
    val: Union[float, int, torch.Tensor, FreeParam],
    name: str = "param",
) -> nn.Parameter:
    """Convert a float, tensor, or FreeParam into an nn.Parameter with gradient tracking."""
    if isinstance(val, FreeParam):
        p = nn.Parameter(torch.tensor(float(val.default_val), dtype=torch.float32))
        setattr(p, "_free_param", val)
        return p
    if isinstance(val, nn.Parameter):
        return val
    if isinstance(val, torch.Tensor):
        if val.requires_grad:
            return nn.Parameter(val)
        return nn.Parameter(val.clone().detach().to(torch.float32))
    if isinstance(val, (int, float)):
        return nn.Parameter(torch.tensor(float(val), dtype=torch.float32))
    raise TypeError(f"Cannot initialize parameter '{name}' from type {type(val)}")


class SineOscillator(nn.Module):
    """Differentiable sinusoidal oscillator."""

    def __init__(
        self,
        freq: Union[float, torch.Tensor, FreeParam] = 440.0,
        phase: Union[float, torch.Tensor, FreeParam] = 0.0,
    ) -> None:
        super().__init__()
        self.freq = _ensure_param(freq, "freq")
        self.phase = _ensure_param(phase, "phase")

    def forward(
        self,
        t: torch.Tensor,
        freq: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute sine wave over time tensor t."""
        f = self.freq if freq is None else freq
        phi = self.phase if phase is None else phase
        sig = torch.sin(2.0 * math.pi * f * t + phi)
        return torch.clamp(sig, -1.0, 1.0)


class SawOscillator(nn.Module):
    """Differentiable sawtooth oscillator."""

    def __init__(
        self,
        freq: Union[float, torch.Tensor, FreeParam] = 440.0,
        phase: Union[float, torch.Tensor, FreeParam] = 0.0,
    ) -> None:
        super().__init__()
        self.freq = _ensure_param(freq, "freq")
        self.phase = _ensure_param(phase, "phase")

    def forward(
        self,
        t: torch.Tensor,
        freq: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute sawtooth wave over time tensor t."""
        f = self.freq if freq is None else freq
        phi = self.phase if phase is None else phase
        p = f * t + phi / (2.0 * math.pi)
        sig = 2.0 * (p - torch.floor(p + 0.5))
        return torch.clamp(sig, -1.0, 1.0)


class SquareOscillator(nn.Module):
    """Differentiable square / pulse wave oscillator with variable duty cycle."""

    def __init__(
        self,
        freq: Union[float, torch.Tensor, FreeParam] = 440.0,
        duty: Union[float, torch.Tensor, FreeParam] = 0.5,
        phase: Union[float, torch.Tensor, FreeParam] = 0.0,
        sharpness: float = 15.0,
    ) -> None:
        super().__init__()
        self.freq = _ensure_param(freq, "freq")
        self.duty = _ensure_param(duty, "duty")
        self.phase = _ensure_param(phase, "phase")
        self.sharpness = float(sharpness)

    def forward(
        self,
        t: torch.Tensor,
        freq: Optional[torch.Tensor] = None,
        duty: Optional[torch.Tensor] = None,
        phase: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute differentiable square wave over time tensor t."""
        f = self.freq if freq is None else freq
        d = self.duty if duty is None else duty
        phi = self.phase if phase is None else phase

        p = f * t + phi / (2.0 * math.pi)
        thresh = torch.sin(math.pi * (0.5 - d))
        sig = torch.tanh(self.sharpness * (torch.sin(2.0 * math.pi * p) - thresh))
        return torch.clamp(sig, -1.0, 1.0)


class NoiseGenerator(nn.Module):
    """Deterministic seeded or random noise generator."""

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        self.seed = seed

    def forward(
        self,
        t: torch.Tensor,
        noise_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate noise matching time tensor t shape."""
        if noise_tensor is not None:
            return torch.clamp(noise_tensor, -1.0, 1.0)
        if self.seed is not None:
            g = torch.Generator(device=t.device).manual_seed(self.seed)
            noise = torch.rand(t.shape, generator=g, device=t.device, dtype=t.dtype) * 2.0 - 1.0
        else:
            noise = torch.rand(t.shape, device=t.device, dtype=t.dtype) * 2.0 - 1.0
        return torch.clamp(noise, -1.0, 1.0)


class ADSREnvelope(nn.Module):
    """Differentiable ADSR (Attack, Decay, Sustain, Release) envelope."""

    def __init__(
        self,
        attack: Union[float, torch.Tensor, FreeParam] = 0.01,
        decay: Union[float, torch.Tensor, FreeParam] = 0.05,
        sustain: Union[float, torch.Tensor, FreeParam] = 0.7,
        release: Union[float, torch.Tensor, FreeParam] = 0.1,
    ) -> None:
        super().__init__()
        self.attack = _ensure_param(attack, "attack")
        self.decay = _ensure_param(decay, "decay")
        self.sustain = _ensure_param(sustain, "sustain")
        self.release = _ensure_param(release, "release")

    def forward(
        self,
        t: torch.Tensor,
        gate_duration: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute ADSR envelope over time tensor t."""
        att = torch.clamp(self.attack, min=1e-4)
        dec = torch.clamp(self.decay, min=1e-4)
        sus = torch.clamp(self.sustain, 0.0, 1.0)
        rel = torch.clamp(self.release, min=1e-4)

        if gate_duration is None:
            max_t = t.max().item() if t.numel() > 0 else 1.0
            gate = torch.tensor(0.7 * max_t, device=t.device, dtype=t.dtype)
        elif isinstance(gate_duration, (int, float)):
            gate = torch.tensor(float(gate_duration), device=t.device, dtype=t.dtype)
        else:
            gate = gate_duration

        t_att = att
        t_dec = t_att + dec

        env_attack = t / t_att
        env_decay = 1.0 - (1.0 - sus) * ((t - t_att) / dec)
        env_sustain = sus.expand_as(t)

        t_rel = t - gate
        env_release = sus * (1.0 - (t_rel / rel))

        pre_gate = torch.where(
            t < t_att,
            env_attack,
            torch.where(t < t_dec, env_decay, env_sustain),
        )
        full_env = torch.where(t < gate, pre_gate, env_release)
        return torch.clamp(full_env, 0.0, 1.0)


class ExpDecayEnvelope(nn.Module):
    """Exponential decay envelope: e(t) = exp(-lambda * max(t, 0))."""

    def __init__(
        self,
        decay_rate: Union[float, torch.Tensor, FreeParam] = 5.0,
    ) -> None:
        super().__init__()
        self.decay_rate = _ensure_param(decay_rate, "decay_rate")

    def forward(
        self,
        t: torch.Tensor,
        decay_rate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute exponential decay over time tensor t."""
        rate = self.decay_rate if decay_rate is None else decay_rate
        rate_clamped = torch.clamp(rate, min=0.0)
        env = torch.exp(-rate_clamped * torch.clamp(t, min=0.0))
        return torch.clamp(env, 0.0, 1.0)


class LFO(nn.Module):
    """Low Frequency Oscillator for dynamic parameter modulation."""

    def __init__(
        self,
        freq: Union[float, torch.Tensor, FreeParam] = 1.0,
        depth: Union[float, torch.Tensor, FreeParam] = 1.0,
        waveform: str = "sine",
    ) -> None:
        super().__init__()
        self.freq = _ensure_param(freq, "freq")
        self.depth = _ensure_param(depth, "depth")
        self.waveform = str(waveform).lower()

    def forward(
        self,
        t: torch.Tensor,
        freq: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute LFO signal over time tensor t."""
        f = self.freq if freq is None else freq
        d = self.depth if depth is None else depth

        if self.waveform == "sine":
            base = torch.sin(2.0 * math.pi * f * t)
        elif self.waveform == "saw":
            p = f * t
            base = 2.0 * (p - torch.floor(p + 0.5))
        elif self.waveform == "square":
            base = torch.tanh(15.0 * torch.sin(2.0 * math.pi * f * t))
        elif self.waveform == "triangle":
            p = f * t
            base = 2.0 * torch.abs(2.0 * (p - torch.floor(p + 0.5))) - 1.0
        else:
            raise ValueError(f"Unsupported LFO waveform: {self.waveform}")

        return torch.clamp(d * base, -1.0, 1.0)


class FMOperator(nn.Module):
    """Frequency Modulation (FM) synthesis operator."""

    def __init__(
        self,
        carrier_freq: Union[float, torch.Tensor, FreeParam] = 440.0,
        mod_freq: Union[float, torch.Tensor, FreeParam] = 220.0,
        mod_depth: Union[float, torch.Tensor, FreeParam] = 1.0,
    ) -> None:
        super().__init__()
        self.carrier_freq = _ensure_param(carrier_freq, "carrier_freq")
        self.mod_freq = _ensure_param(mod_freq, "mod_freq")
        self.mod_depth = _ensure_param(mod_depth, "mod_depth")

    def forward(
        self,
        t: torch.Tensor,
        carrier_freq: Optional[torch.Tensor] = None,
        mod_freq: Optional[torch.Tensor] = None,
        mod_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute 2-operator FM synthesis: sin(2*pi*fc*t + I * sin(2*pi*fm*t))."""
        fc = self.carrier_freq if carrier_freq is None else carrier_freq
        fm = self.mod_freq if mod_freq is None else mod_freq
        i = self.mod_depth if mod_depth is None else mod_depth

        modulator = i * torch.sin(2.0 * math.pi * fm * t)
        sig = torch.sin(2.0 * math.pi * fc * t + modulator)
        return torch.clamp(sig, -1.0, 1.0)


class FMSynthesisPair(FMOperator):
    """Alias for 2-operator FM synthesis pair."""

    pass


class LowPassFilter(nn.Module):
    """Differentiable frequency-domain lowpass filter."""

    def __init__(
        self,
        cutoff: Union[float, torch.Tensor, FreeParam] = 1000.0,
        resonance: Union[float, torch.Tensor, FreeParam] = 1.0,
        sample_rate: float = 16000.0,
        order: int = 2,
    ) -> None:
        super().__init__()
        self.cutoff = _ensure_param(cutoff, "cutoff")
        self.resonance = _ensure_param(resonance, "resonance")
        self.sample_rate = float(sample_rate)
        self.order = int(order)

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        cutoff: Optional[torch.Tensor] = None,
        resonance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Filter audio signal x."""
        fc = self.cutoff if cutoff is None else cutoff
        q = self.resonance if resonance is None else resonance

        fs = self.sample_rate
        if t is not None and t.numel() > 1:
            dt = (t[..., 1] - t[..., 0]).mean().item()
            if dt > 0.0:
                fs = 1.0 / dt

        n = x.shape[-1]
        x_fft = torch.fft.rfft(x, dim=-1)
        freqs = torch.fft.rfftfreq(n, d=1.0 / fs, device=x.device)

        fc_safe = torch.clamp(fc, min=10.0)
        q_safe = torch.clamp(q, min=0.1)
        omega = freqs / fc_safe

        if self.order == 1:
            denom = torch.complex(torch.ones_like(omega), omega)
        else:
            denom = torch.complex(1.0 - omega**2, omega / q_safe)

        h = 1.0 / denom
        y_fft = x_fft * h
        return torch.fft.irfft(y_fft, n=n, dim=-1)


class GainModule(nn.Module):
    """Differentiable gain scaling module."""

    def __init__(
        self,
        gain: Union[float, torch.Tensor, FreeParam] = 1.0,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        self.gain = _ensure_param(gain, "gain")
        self.clamp_output = bool(clamp_output)

    def forward(
        self,
        x: torch.Tensor,
        gain: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scale signal by gain."""
        g = self.gain if gain is None else gain
        out = g * x
        if self.clamp_output:
            return torch.clamp(out, -1.0, 1.0)
        return out


class MixModule(nn.Module):
    """Differentiable linear mixing module."""

    def __init__(
        self,
        ratio: Union[float, torch.Tensor, FreeParam] = 0.5,
    ) -> None:
        super().__init__()
        self.ratio = _ensure_param(ratio, "ratio")

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        ratio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mix two signals linearly: (1 - r) * x1 + r * x2."""
        r = self.ratio if ratio is None else ratio
        r_clamped = torch.clamp(r, 0.0, 1.0)
        mixed = (1.0 - r_clamped) * x1 + r_clamped * x2
        return torch.clamp(mixed, -1.0, 1.0)
