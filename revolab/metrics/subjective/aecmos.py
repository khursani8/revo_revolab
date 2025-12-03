"""
AEC-MOS: Acoustic Echo Cancellation Mean Opinion Score.
"""
from typing import Union, Dict, Any
from ..base import AudioMetric

try:
    from speechmos import aecmos
except ImportError:
    aecmos = None

class AECMOS(AudioMetric):
    """
    AEC-MOS Score Metric.

    Uses a pre-trained model to predict the AEC-MOS score.
    """
    def __init__(self, sample_rate: int = 16000, **kwargs):
        """
        Initialize AEC-MOS metric.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if aecmos is None:
            raise ImportError("Please install speechmos to use this metric.")
        
        if self.sample_rate != 16000:
            print(f"Warning: The AEC-MOS model expects 16kHz audio. Audio will be resampled.")

    def compute(
                self,
                farend_audio: Union[str, Any],
                mic_audio: Union[str, Any],
                enhanced_audio: Union[str, Any]) -> Dict[str, Any]:
        """
        Compute the AEC-MOS score.
        """
        # speechmos expects file paths
        def to_path(audio):
            if isinstance(audio, str):
                return audio
            import soundfile as sf
            import tempfile
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(f.name, audio, self.sample_rate)
            return f.name

        farend_path = to_path(farend_audio)
        mic_path = to_path(mic_audio)
        enhanced_path = to_path(enhanced_audio)

        scores = aecmos.run(farend_path, mic_path, enhanced_path)
        return scores


def aecmos_score(
                 farend_audio: Union[str, Any],
                 mic_audio: Union[str, Any],
                 enhanced_audio: Union[str, Any],
                 **kwargs) -> Dict[str, Any]:
    """
    Convenience function to compute AEC-MOS score.
    """
    metric = AECMOS(**kwargs)
    return metric.compute(farend_audio, mic_audio, enhanced_audio)


if __name__ == '__main__':
    print("=== AEC-MOS Score Demo ===")
    from revolab.metrics.base import create_demo_audio_data
    import soundfile as sf
    import tempfile
    
    print("Initializing metric (may download models)...")
    metric = AECMOS()
    
    demo_data = create_demo_audio_data()
    
    # AEC-MOS requires three audio files, we will use the demo data as placeholders
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f1, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f2, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f3:
        
        sf.write(f1.name, demo_data['clean'], 16000)
        sf.write(f2.name, demo_data['noisy'], 16000)
        sf.write(f3.name, demo_data['clean'], 16000)
        
        print("Computing score...")
        scores = metric.compute(f1.name, f2.name, f3.name)
        print(f"AEC-MOS Scores: {scores}")
