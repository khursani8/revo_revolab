"""
UT-MOS: Universal Speech Quality Prediction Metric.
"""
from typing import Union, Dict, Any
import numpy as np
from ..base import AudioMetric

try:
    import torch
except ImportError:
    torch = None

class UTMOS(AudioMetric):
    """
    UT-MOS Score Metric.

    Uses a pre-trained model from torch.hub to predict the subjective
    Mean Opinion Score (MOS) for speech quality. This implementation uses the
    `tarepan/SpeechMOS:v1.2.0` model.
    """
    def __init__(self, sample_rate: int = 16000, model_name: str = "utmos22_strong", repo: str = "tarepan/SpeechMOS:v1.2.0", trust_repo: bool = True, predictor=None, **kwargs):
        """
        Initialize UT-MOS metric.
        
        Args:
            sample_rate (int): The sample rate of the audio. The model expects 16kHz.
            model_name (str): The model name to load from the hub.
            repo (str): The repository from which to load the model.
            trust_repo (bool): Whether to trust the repository code.
            predictor: An optional pre-initialized predictor object.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if torch is None:
            raise ImportError("Please install torch to use the UTMOS metric.")
        
        if self.sample_rate != 16000:
            print(f"Warning: The UT-MOS model expects 16kHz audio, but sample_rate is {self.sample_rate}. The audio will be resampled by the audio loader.")

        if predictor:
            self.predictor = predictor
        else:
            # Download the model once to cache it
            self.predictor = torch.hub.load(repo, model_name, trust_repo=trust_repo)

    def compute(self, audio: Union[str, np.ndarray]) -> float:
        """
        Compute the UT-MOS score.
        
        Args:
            audio (Union[str, np.ndarray]): The audio to score (file path or numpy array).

        Returns:
            float: The predicted MOS score.
        """
        if isinstance(audio, str):
            # The model expects a 16kHz sample rate
            audio = self.load_audio(audio, sr=16000)

        # Normalize audio if it's not already in [-1, 1]
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))

        # Convert to torch tensor
        wave_tensor = torch.from_numpy(audio).unsqueeze(0).float()

        # Predict the MOS score
        score = self.predictor(wave_tensor, self.sample_rate)
        
        return score.item()

def utmos(audio: Union[str, np.ndarray], **kwargs) -> float:
    """
    Convenience function to compute UT-MOS score.
    """
    metric = UTMOS(**kwargs)
    return metric.compute(audio)


if __name__ == '__main__':
    print("=== UT-MOS Score Demo ===")
    from revolab.metrics.base import create_demo_audio_data
    
    print("Initializing metric (may download models)...")
    metric = UTMOS()
    
    demo_data = create_demo_audio_data()
    
    print("Computing score for noisy audio...")
    score = metric.compute(demo_data['noisy'])
    print(f"UT-MOS Score: {score:.4f}")
    
    print("Computing score for clean audio...")
    score_clean = metric.compute(demo_data['clean'])
    print(f"UT-MOS Score (Clean): {score_clean:.4f}")