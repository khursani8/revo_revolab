"""
Signal to Interference Ratio (SIR) metric implementation.
"""

import numpy as np
from typing import Union

from ..base import AudioMetric


class SIR(AudioMetric):
    """
    Signal to Interference Ratio metric.
    """

    def __init__(self, sample_rate: int = 16000, **kwargs):
        """
        Initialize SIR metric.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

    def _compute_sir(self,
                     reference: np.ndarray,
                     interference: np.ndarray) -> float:
        """
        Compute SIR between a reference and an interference signal.
        """
        min_len = min(len(reference), len(interference))
        reference = reference[:min_len]
        interference = interference[:min_len]

        reference_energy = np.sum(reference ** 2)
        interference_energy = np.sum(interference ** 2)

        if interference_energy < 1e-8:
            return np.inf

        sir = 10 * np.log10(reference_energy / interference_energy)
        return float(sir)

    def compute(self,
                ref_audio: Union[str, np.ndarray],
                interference_audio: Union[str, np.ndarray]) -> float:
        """
        Compute SIR.
        
        Args:
            ref_audio: The clean reference signal.
            interference_audio: The interference signal present in the estimation.
        """
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(interference_audio, str):
            interference_audio = self.load_audio(interference_audio)

        ref_audio = self.preprocess_audio(ref_audio)
        interference_audio = self.preprocess_audio(interference_audio)

        return self._compute_sir(ref_audio, interference_audio)


def sir(ref_audio: Union[str, np.ndarray],
        interference_audio: Union[str, np.ndarray],
        **kwargs) -> float:
    """
    Convenience function to compute SIR.
    """
    metric = SIR(**kwargs)
    return metric.compute(ref_audio, interference_audio)

if __name__ == '__main__':
    print("=== SIR (Signal to Interference Ratio) Demo ===")
    from revolab.metrics.base import create_demo_audio_data

    sir_metric = SIR()
    demo_data = create_demo_audio_data()
    ref_audio = demo_data['clean']
    
    # Create a simple interference signal (e.g., another sine wave)
    sr = demo_data['sample_rate']
    t = np.linspace(0, len(ref_audio) / sr, len(ref_audio), False)
    interference = 0.5 * np.sin(2 * np.pi * 500 * t)
    interference = interference.astype(np.float32)

    sir_value = sir(ref_audio, interference)
    print(f"Clean vs. Interference: SIR = {sir_value:.2f} dB")

    # Interference with lower power
    interference_low = 0.1 * interference
    sir_value_low = sir(ref_audio, interference_low)
    print(f"Clean vs. Low Power Interference: SIR = {sir_value_low:.2f} dB")
