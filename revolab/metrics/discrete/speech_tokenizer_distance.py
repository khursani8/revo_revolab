"""
Speech Tokenizer Distance Metric.

This metric uses a speech tokenizer (SNAC) to convert audio into a sequence
of discrete tokens, and then computes the Token Error Rate (TER) between the
reference and estimated token sequences.
"""

from typing import Union, Dict, Any
import torch
from ..base import AudioMetric

try:
    from snac import SNAC
    from jiwer import process_words
except ImportError:
    SNAC = None
    process_words = None

class SpeechTokenizerDistance(AudioMetric):
    """
    Speech Tokenizer Distance Metric.

    Tokenizes audio using SNAC and computes the Token Error Rate (TER).
    """
    def __init__(self, sample_rate: int = 44100, model_name: str = "hubert_base", tokenizer=None, **kwargs):
        """
        Initialize SpeechTokenizerDistance metric.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if SNAC is None or process_words is None:
            raise ImportError("Please install snac and jiwer to use this metric.")
        
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SNAC.from_pretrained(f"babysor/snac-{self.sample_rate}-{model_name}")

    def compute(self,
                ref_audio: Union[str, Any],
                est_audio: Union[str, Any]) -> Dict[str, Any]:
        """
        Compute Speech Tokenizer Distance metrics.
        """
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)

        # SNAC expects a batch dimension and a channel dimension
        ref_tokens = self.tokenizer.encode(torch.from_numpy(ref_audio).unsqueeze(0).unsqueeze(0))
        est_tokens = self.tokenizer.encode(torch.from_numpy(est_audio).unsqueeze(0).unsqueeze(0))

        # The tokens are integers, convert them to strings for jiwer
        ref_token_str = " ".join(map(str, ref_tokens[0][0].tolist()))
        est_token_str = " ".join(map(str, est_tokens[0][0].tolist()))
        
        results = process_words(ref_token_str, est_token_str)

        return {
            "ter": results.wer, # Token Error Rate is equivalent to WER here
            "token_substitutions": results.substitutions,
            "token_deletions": results.deletions,
            "token_insertions": results.insertions,
        }

def speech_tokenizer_distance(ref_audio: Union[str, Any],
                                est_audio: Union[str, Any],
                                **kwargs) -> float:
    """
    Convenience function to compute Speech Tokenizer Distance.
    """
    metric = SpeechTokenizerDistance(**kwargs)
    return metric.compute(ref_audio, est_audio)["ter"]


if __name__ == '__main__':
    print("=== Speech Tokenizer Distance Demo ===")
    from revolab.metrics.base import create_demo_audio_data
    
    # SNAC models are available for 24k, 32k, and 44.1k. We'll use 44.1k.
    print("Initializing metric (may download models)...")
    metric = SpeechTokenizerDistance(sample_rate=44100)
    
    # The demo audio is 16k, so this will be upsampled by the loader.
    # The results won't be meaningful, but it demonstrates functionality.
    demo_data = create_demo_audio_data()
    
    print("Computing score...")
    scores = metric.compute(demo_data['clean'], demo_data['noisy'])
    
    print(f"Token Error Rate (TER): {scores['ter']:.4f}")
