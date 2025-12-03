"""
Signal to Artifact Ratio (SAR) metric implementation.
"""

import numpy as np
from typing import Union

from ..base import AudioMetric


class SAR(AudioMetric):
    """
    Signal to Artifact Ratio metric.
    
    Note: This is a simplified implementation of SAR. A more accurate
    calculation would require a full BSS Eval-style decomposition of the
    signal, which is beyond the scope of this basic metric.
    """

    def __init__(self, sample_rate: int = 16000, **kwargs):
        """
        Initialize SAR metric.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

    def _compute_sar(self,
                     reference: np.ndarray,
                     estimate: np.ndarray) -> float:
        """
        Compute SAR between a reference and an estimate signal.
        """
        min_len = min(len(reference), len(estimate))
        reference = reference[:min_len]
        estimate = estimate[:min_len]

        artifacts = estimate - reference

        reference_energy = np.sum(reference ** 2)
        artifacts_energy = np.sum(artifacts ** 2)

        if artifacts_energy < 1e-8:
            return np.inf

        sar = 10 * np.log10(reference_energy / artifacts_energy)
        return float(sar)

    def compute(self,
                ref_audio: Union[str, np.ndarray],
                est_audio: Union[str, np.ndarray]) -> float:
        """
        Compute SAR.
        """
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)

        ref_audio = self.preprocess_audio(ref_audio)
        est_audio = self.preprocess_audio(est_audio)

        return self._compute_sar(ref_audio, est_audio)


def sar(ref_audio: Union[str, np.ndarray],
        est_audio: Union[str, np.ndarray],
        **kwargs) -> float:
    """
    Convenience function to compute SAR.
    """
    metric = SAR(**kwargs)
    return metric.compute(ref_audio, est_audio)

if __name__ == '__main__':
    print("=== SAR (Signal to Artifact Ratio) Demo ===")
    from revolab.metrics.base import create_demo_audio_data

    sar_metric = SAR()
    demo_data = create_demo_audio_data()
    ref_audio = demo_data['clean']
    est_audio = demo_data['noisy']

    sar_value = sar(ref_audio, est_audio)
    print(f"Clean vs. Noisy: SAR = {sar_value:.2f} dB")
    
    # an almost perfect signal should have high SAR
    almost_perfect = ref_audio + np.random.randn(len(ref_audio)) * 1e-4
    sar_value_high = sar(ref_audio, almost_perfect)
    print(f"Clean vs. Almost Perfect: SAR = {sar_value_high:.2f} dB")

