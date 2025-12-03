"""
Revolab: Comprehensive Speech Evaluation Library.

Revolab provides a wide range of speech evaluation metrics organized into
logical categories for easy access and comprehensive speech analysis.

Quick Usage Examples:

# Import specific metrics
from revolab.metrics import mcd, f0_corr, f0_rmse, sdr

# Import by category
from revolab.metrics.f0_mcd import mcd, f0_corr, f0_rmse
from revolab.metrics.signal import sdr

# Use metrics with file paths
mcd_score = mcd('reference.wav', 'estimated.wav')
sdr_score = sdr('clean.wav', 'enhanced.wav')

# Use metrics with audio arrays
import librosa
ref_audio = librosa.load('reference.wav', sr=16000)[0]
est_audio = librosa.load('estimated.wav', sr=16000)[0]
f0_score = f0_corr(ref_audio, est_audio)

# Run demos
uv run python -m revolab.metrics.f0_mcd.mcd
uv run python -m revolab.metrics.signal.sdr

Available Categories:
- F0 and Mel-Cepstral: mcd, f0_corr, f0_rmse
- Signal Quality: sdr, sir, sar, si_sdr, si_snri
- Quality Metrics: pesq, stoi, visqol
- Discrete Speech: speech_bert, speech_bleu, speech_token_distance
- Subjective: utmos, dnsmos, plcmos, aecmos
"""

# Core metrics (most commonly used)
from .metrics import (
    mcd,
    f0_corr, 
    f0_rmse,
    sdr
)

# Version
__version__ = '0.1.0'
__author__ = 'Revolab Team'
__email__ = 'contact@revolab.ai'

__all__ = [
    # Core metrics
    'mcd',
    'f0_corr',
    'f0_rmse', 
    'sdr',
    
    # Version info
    '__version__',
    '__author__',
    '__email__'
]
