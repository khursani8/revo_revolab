"""
PLC-MOS: Packet Loss Concealment Mean Opinion Score.
"""
from typing import Union, Dict, Any
from ..base import AudioMetric

try:
    from speechmos import plcmos
except ImportError:
    plcmos = None

class PLCMOS(AudioMetric):
    """
    PLC-MOS Score Metric.

    Uses a pre-trained model to predict the PLC-MOS score.
    """
    def __init__(self, sample_rate: int = 16000, **kwargs):
        """
        Initialize PLC-MOS metric.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if plcmos is None:
            raise ImportError("Please install speechmos to use this metric.")
        
        if self.sample_rate != 16000:
            print(f"Warning: The PLC-MOS model expects 16kHz audio. Audio will be resampled.")

    def compute(self,
                audio: Union[str, Any]) -> Dict[str, Any]:
        """
        Compute the PLC-MOS score.
        """
        if isinstance(audio, str):
            pass
        else:
            import soundfile as sf
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                sf.write(f.name, audio, self.sample_rate)
                scores = plcmos.run(f.name)
                return scores

        scores = plcmos.run(audio)
        return scores


def plcmos_score(audio: Union[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to compute PLC-MOS score.
    """
    metric = PLCMOS(**kwargs)
    return metric.compute(audio)


if __name__ == '__main__':
    print("=== PLC-MOS Score Demo ===")
    from revolab.metrics.base import create_demo_audio_data
    import soundfile as sf
    import tempfile
    
    print("Initializing metric (may download models)...")
    metric = PLCMOS()
    
    demo_data = create_demo_audio_data()
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, demo_data['clean'], 16000)
        
        print("Computing score for clean audio...")
        scores = metric.compute(f.name)
        print(f"PLC-MOS Scores: {scores}")

