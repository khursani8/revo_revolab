"""
Short-Time Objective Intelligibility (STOI) metric.
"""
from typing import Union
import numpy as np
from pystoi import stoi as run_stoi
from ..base import AudioMetric

class STOI(AudioMetric):
    """
    Short-Time Objective Intelligibility (STOI) metric.
    
    This metric is a wrapper around the `pystoi` library. It computes the
    STOI score between a clean and a degraded audio signal.
    
    STOI scores range from 0 to 1, where higher scores indicate
    better intelligibility.
    """

    def __init__(self, sample_rate: int = 16000, extended: bool = False, **kwargs):
        """
        Initialize STOI metric.
        
        Args:
            sample_rate: The sample rate of the audio. Must be 10000 or greater.
            extended: Whether to use the extended STOI version.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if self.sample_rate < 10000:
            raise ValueError("STOI requires a sample rate of 10000 or greater.")
        self.extended = extended

    def compute(self,
                ref_audio: Union[str, np.ndarray],
                deg_audio: Union[str, np.ndarray]) -> float:
        """
        Compute the STOI score.
        
        Args:
            ref_audio: The reference (clean) audio (file path or array).
            deg_audio: The degraded audio (file path or array).
        """
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(deg_audio, str):
            deg_audio = self.load_audio(deg_audio)

        return run_stoi(ref_audio, deg_audio, self.sample_rate, self.extended)


def stoi(ref_audio: Union[str, np.ndarray],
         deg_audio: Union[str, np.ndarray],
         sample_rate: int = 16000,
         extended: bool = False,
         **kwargs) -> float:
    """
    Convenience function to compute STOI.
    """
    metric = STOI(sample_rate=sample_rate, extended=extended, **kwargs)
    return metric.compute(ref_audio, deg_audio)


if __name__ == '__main__':
    print("=== STOI (Short-Time Objective Intelligibility) Demo ===")
    from revolab.metrics.base import create_demo_audio_data

    metric = STOI(sample_rate=16000)
    
    demo_data = create_demo_audio_data()
    ref_audio = demo_data['clean']
    deg_audio = demo_data['noisy']

    score = metric.compute(ref_audio, deg_audio)
    print(f"STOI score: {score:.4f}")

    # Test extended STOI
    metric_extended = STOI(sample_rate=16000, extended=True)
    score_extended = metric_extended.compute(ref_audio, deg_audio)
    print(f"Extended STOI score: {score_extended:.4f}")
