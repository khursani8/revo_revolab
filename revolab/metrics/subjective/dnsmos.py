"""
DNS-MOS: Deep Noise Suppression Mean Opinion Score.
"""
from typing import Union, Dict, Any
from ..base import AudioMetric

try:
    from speechmos import dnsmos
except ImportError:
    dnsmos = None

class DNSMOS(AudioMetric):
    """
    DNS-MOS Score Metric.

    Uses a pre-trained model to predict the DNS-MOS score.
    """
    def __init__(self, sample_rate: int = 16000, **kwargs):
        """
        Initialize DNS-MOS metric.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if dnsmos is None:
            raise ImportError("Please install speechmos to use this metric.")
        
        if self.sample_rate != 16000:
            print(f"Warning: The DNS-MOS model expects 16kHz audio. Audio will be resampled.")

    def compute(self,
                audio: Union[str, Any]) -> Dict[str, Any]:
        """
        Compute the DNS-MOS score.
        """
        if isinstance(audio, str):
            # speechmos expects a file path, so we don't load the audio ourselves
            pass
        else:
            # If we have an array, we need to save it to a temporary file
            import soundfile as sf
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio, self.sample_rate)
                scores = dnsmos.run(f.name)
                return scores

        scores = dnsmos.run(audio)
        return scores


def dnsmos_score(audio: Union[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to compute DNS-MOS score.
    """
    metric = DNSMOS(**kwargs)
    return metric.compute(audio)


if __name__ == '__main__':
    print("=== DNS-MOS Score Demo ===")
    from revolab.metrics.base import create_demo_audio_data
    import soundfile as sf
    import tempfile
    
    print("Initializing metric (may download models)...")
    metric = DNSMOS()
    
    demo_data = create_demo_audio_data()
    
    # Save the demo audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, demo_data['noisy'], 16000)
        
        print("Computing score for noisy audio...")
        scores = metric.compute(f.name)
        print(f"DNS-MOS Scores: {scores}")

