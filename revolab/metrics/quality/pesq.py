"""
Perceptual Evaluation of Speech Quality (PESQ) metric.
"""
from typing import Union
import numpy as np
from pesq import pesq as run_pesq
from ..base import AudioMetric

class PESQ(AudioMetric):
    """
    Perceptual Evaluation of Speech Quality (PESQ) metric.
    
    This metric is a wrapper around the `pesq` library. It computes the
    PESQ score between a reference and a degraded audio signal.
    
    PESQ scores range from -0.5 to 4.5, where higher scores indicate
    better quality.
    """

    def __init__(self, sample_rate: int = 16000, mode: str = 'wb', **kwargs):
        """
        Initialize PESQ metric.
        
        Args:
            sample_rate: The sample rate of the audio. Must be 8000 or 16000.
            mode: 'wb' for wide-band (16kHz) or 'nb' for narrow-band (8kHz).
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if self.sample_rate not in [8000, 16000]:
            raise ValueError("PESQ only supports sample rates of 8000 or 16000.")
        if mode not in ['wb', 'nb']:
            raise ValueError("PESQ mode must be 'wb' or 'nb'.")
        if mode == 'wb' and self.sample_rate == 8000:
            raise ValueError("Wide-band mode requires a sample rate of 16000.")
        if mode == 'nb' and self.sample_rate == 16000:
            # The pesq library can handle this, but it's better to be explicit
            print("Warning: Narrow-band PESQ with 16kHz audio. The audio will be downsampled.")

        self.mode = mode

    def compute(self,
                ref_audio: Union[str, np.ndarray],
                deg_audio: Union[str, np.ndarray]) -> float:
        """
        Compute the PESQ score.
        
        Args:
            ref_audio: The reference audio (file path or array).
            deg_audio: The degraded audio (file path or array).
        """
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(deg_audio, str):
            deg_audio = self.load_audio(deg_audio)

        # The pesq library expects float32 arrays
        ref_audio = ref_audio.astype(np.float32)
        deg_audio = deg_audio.astype(np.float32)

        return run_pesq(self.sample_rate, ref_audio, deg_audio, self.mode)


def pesq(ref_audio: Union[str, np.ndarray],
         deg_audio: Union[str, np.ndarray],
         sample_rate: int = 16000,
         mode: str = 'wb',
         **kwargs) -> float:
    """
    Convenience function to compute PESQ.
    """
    metric = PESQ(sample_rate=sample_rate, mode=mode, **kwargs)
    return metric.compute(ref_audio, deg_audio)


if __name__ == '__main__':
    print("=== PESQ (Perceptual Evaluation of Speech Quality) Demo ===")
    from revolab.metrics.base import create_demo_audio_data

    # PESQ requires 16kHz or 8kHz
    metric = PESQ(sample_rate=16000, mode='wb')
    
    demo_data = create_demo_audio_data()
    ref_audio = demo_data['clean']
    deg_audio = demo_data['noisy']

    score = metric.compute(ref_audio, deg_audio)
    print(f"PESQ score (wide-band): {score:.2f}")

    # Test narrow-band
    metric_nb = PESQ(sample_rate=16000, mode='nb')
    score_nb = metric_nb.compute(ref_audio, deg_audio)
    print(f"PESQ score (narrow-band, downsampled): {score_nb:.2f}")

