"""
Scale-Invariant Signal-to-Noise Ratio Improvement (SI-SNRI) metric.
"""

import numpy as np
from typing import Union

from ..base import AudioMetric
from .si_sdr import SISDR


class SISNRI(AudioMetric):
    """
    Scale-Invariant Signal-to-Noise Ratio Improvement (SI-SNRI) metric.
    
    This metric evaluates the improvement in signal-to-noise ratio between
    a noisy signal and an enhanced signal, relative to a clean reference.
    It is scale-invariant, meaning it is not affected by the absolute
    amplitude of the signals.
    """

    def __init__(self, sample_rate: int = 16000, **kwargs):
        """
        Initialize SI-SNRI metric.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._si_sdr_metric = SISDR(sample_rate=sample_rate, **kwargs)

    def compute(self,
                clean_audio: Union[str, np.ndarray],
                noisy_audio: Union[str, np.ndarray],
                enhanced_audio: Union[str, np.ndarray]) -> float:
        """
        Compute SI-SNRI.
        
        Args:
            clean_audio: The clean reference signal.
            noisy_audio: The noisy (unprocessed) signal.
            enhanced_audio: The enhanced (processed) signal.
        """
        if isinstance(clean_audio, str):
            clean_audio = self.load_audio(clean_audio)
        if isinstance(noisy_audio, str):
            noisy_audio = self.load_audio(noisy_audio)
        if isinstance(enhanced_audio, str):
            enhanced_audio = self.load_audio(enhanced_audio)

        clean_audio = self.preprocess_audio(clean_audio)
        noisy_audio = self.preprocess_audio(noisy_audio)
        enhanced_audio = self.preprocess_audio(enhanced_audio)

        si_snr_enhanced = self._si_sdr_metric.compute(clean_audio, enhanced_audio)
        si_snr_noisy = self._si_sdr_metric.compute(clean_audio, noisy_audio)

        return si_snr_enhanced - si_snr_noisy


def si_snri(clean_audio: Union[str, np.ndarray],
            noisy_audio: Union[str, np.ndarray],
            enhanced_audio: Union[str, np.ndarray],
            **kwargs) -> float:
    """
    Convenience function to compute SI-SNRI.
    """
    metric = SISNRI(**kwargs)
    return metric.compute(clean_audio, noisy_audio, enhanced_audio)


if __name__ == '__main__':
    print("=== SI-SNRI (Scale-Invariant Signal-to-Noise Ratio Improvement) Demo ===")
    from revolab.metrics.base import create_demo_audio_data

    metric = SISNRI()
    demo_data = create_demo_audio_data()
    clean_audio = demo_data['clean']
    noisy_audio = demo_data['noisy']

    # Create a simple "enhanced" audio by reducing the noise
    enhanced_audio = (clean_audio + 0.05 * np.random.randn(len(clean_audio))).astype(np.float32)

    improvement = si_snri(clean_audio, noisy_audio, enhanced_audio)
    print(f"SI-SNRI: {improvement:.2f} dB")
    
    # if enhanced is worse than noisy, improvement should be negative
    worse_audio = (clean_audio + 0.2 * np.random.randn(len(clean_audio))).astype(np.float32)
    negative_improvement = si_snri(clean_audio, noisy_audio, worse_audio)
    print(f"Negative SI-SNRI: {negative_improvement:.2f} dB")
