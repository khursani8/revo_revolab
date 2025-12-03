"""
UT-MOS: Universal Speech Quality Prediction Metric.
"""
from typing import Union, Dict, Any
from ..base import AudioMetric

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class UTMOS(AudioMetric):
    """
    UT-MOS Score Metric.

    Uses a pre-trained model to predict the subjective Mean Opinion Score (MOS)
    for speech quality.
    """
    def __init__(self, sample_rate: int = 16000, model: str = "microsoft/speechmos-utmos-speech-asr", utmos_pipeline=None, **kwargs):
        """
        Initialize UT-MOS metric.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if pipeline is None:
            raise ImportError("Please install torch and transformers to use this metric.")
        
        if self.sample_rate != 16000:
            print(f"Warning: The UT-MOS model expects 16kHz audio. Audio will be resampled.")

        if utmos_pipeline:
            self.utmos_pipeline = utmos_pipeline
        else:
            self.utmos_pipeline = pipeline("audio-classification", model=model)

    def compute(self,
                audio: Union[str, Any]) -> Dict[str, Any]:
        """
        Compute the UT-MOS score.
        """
        if isinstance(audio, str):
            audio = self.load_audio(audio)

        # The pipeline expects a dictionary with 'array' and 'sampling_rate' keys
        input_data = {"array": audio, "sampling_rate": self.sample_rate}
        
        # The returned score is a list of dictionaries
        result = self.utmos_pipeline(input_data, top_k=1)
        
        score = result[0]['score']

        return {
            "score": score
        }

def utmos(audio: Union[str, Any], **kwargs) -> float:
    """
    Convenience function to compute UT-MOS score.
    """
    metric = UTMOS(**kwargs)
    return metric.compute(audio)["score"]


if __name__ == '__main__':
    print("=== UT-MOS Score Demo ===")
    from revolab.metrics.base import create_demo_audio_data
    
    print("Initializing metric (may download models)...")
    metric = UTMOS()
    
    demo_data = create_demo_audio_data()
    
    print("Computing score for noisy audio...")
    scores = metric.compute(demo_data['noisy'])
    print(f"UT-MOS Score: {scores['score']:.4f}")
