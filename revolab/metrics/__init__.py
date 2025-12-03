"""
Revolab Speech Metrics Library.

This module provides comprehensive speech evaluation metrics organized by category:

F0 and Mel-Cepstral Metrics:
- mcd: Mel Cepstral Distortion
- f0_corr: F0 Correlation  
- f0_rmse: F0 Root Mean Square Error

Signal Quality Metrics:
- sdr: Signal to Distortion Ratio
- sir: Signal to Interference Ratio (coming soon)
- sar: Signal to Artifact Ratio (coming soon)
- si_sdr: Scale-Invariant Signal to Distortion Ratio (coming soon)
- si_snri: Scale-Invariant Signal to Noise Ratio Improvement (coming soon)

Quality Metrics:
- pesq: Perceptual Evaluation of Speech Quality (coming soon)
- stoi: Short-Time Objective Intelligibility (coming soon)
- visqol: Virtual Speech Quality Objective Listener (coming soon)

Discrete Speech Metrics:
- speech_bert: Speech BERT Score (coming soon)
- speech_bleu: Speech BLEU Score (coming soon)
- speech_token_distance: Speech Token Distance (coming soon)

Subjective Metrics:
- utmos: UT-MOS Score (coming soon)
- dnsmos: DNS-MOS Score (coming soon)
- plcmos: PLC-MOS Score (coming soon)
- aecmos: AEC-MOS Score (coming soon)
"""

# F0 and Mel-Cepstral metrics
from .f0_mcd import mcd, f0_corr, f0_rmse

# Signal quality metrics
from .signal import sdr

# Base classes and utilities
from .base import BaseMetric, AudioMetric

__all__ = [
    # F0 and Mel-Cepstral metrics
    'mcd',
    'f0_corr', 
    'f0_rmse',
    
    # Signal quality metrics
    'sdr',
    
    # Base classes
    'BaseMetric',
    'AudioMetric'
]

# Version info
__version__ = '0.1.0'
