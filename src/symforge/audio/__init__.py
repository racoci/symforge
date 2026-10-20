"""Audio domain DSP primitives and multi-scale STFT contrastive perceptual loss extractor."""

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
from symforge.audio.stft import MultiScaleSTFTExtractor

__all__ = [
    "SineOscillator",
    "SawOscillator",
    "SquareOscillator",
    "NoiseGenerator",
    "ADSREnvelope",
    "ExpDecayEnvelope",
    "LFO",
    "FMOperator",
    "FMSynthesisPair",
    "LowPassFilter",
    "GainModule",
    "MixModule",
    "MultiScaleSTFTExtractor",
]
