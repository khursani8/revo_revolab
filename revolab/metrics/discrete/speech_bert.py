"""
Speech BERT Score metric.

This metric uses an Automatic Speech Recognition (ASR) model to transcribe
reference and estimated audio, then computes the BERTScore between the
transcribed texts.
"""

from typing import Union, Dict, Any
import numpy as np
from ..base import AudioMetric

# Try to import heavy dependencies, but allow failure for optional installation
try:
    from transformers import pipeline
    from bert_score import score as bert_score_compute
except ImportError:
    pipeline = None
    bert_score_compute = None

class SpeechBERT(AudioMetric):
    """
    Speech BERT Score Metric.

    Transcribes reference and estimated audio using an ASR model and then
    computes the BERTScore (F1) between the transcriptions.
    """
    def __init__(self, sample_rate: int = 16000, asr_model: str = "facebook/wav2vec2-base-960h", asr_pipeline=None, **kwargs):
        """
        Initialize SpeechBERT metric.

        Args:
            sample_rate: Audio sample rate (must be 16000 for the default model).
            asr_model: The ASR model to use from the Hugging Face Hub.
            asr_pipeline: An optional pre-initialized ASR pipeline object.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if pipeline is None or bert_score_compute is None:
            raise ImportError("Please install torch, transformers, and bert-score to use the SpeechBERT metric.")
        
        if asr_pipeline:
            self.asr_pipeline = asr_pipeline
        else:
            self.asr_pipeline = pipeline("automatic-speech-recognition", model=asr_model)

    def compute(self,
                ref_audio: Union[str, np.ndarray],
                est_audio: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Compute the Speech BERT score.

        Args:
            ref_audio: The reference audio (file path or array).
            est_audio: The estimated audio (file path or array).

        Returns:
            A dictionary containing the precision, recall, and F1 BERT scores,
            as well as the transcribed texts.
        """
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)

        # Transcribe audio
        ref_text = self.asr_pipeline(ref_audio)["text"]
        est_text = self.asr_pipeline(est_audio)["text"]

        # Compute BERT score
        P, R, F1 = bert_score_compute([est_text], [ref_text], lang="en", verbose=False)

        return {
            "precision": P.item(),
            "recall": R.item(),
            "f1": F1.item(),
            "ref_transcription": ref_text,
            "est_transcription": est_text,
        }

def speech_bert(ref_audio: Union[str, np.ndarray],
                est_audio: Union[str, np.ndarray],
                **kwargs) -> Dict[str, Any]:
    """
    Convenience function to compute Speech BERT score.
    """
    metric = SpeechBERT(**kwargs)
    return metric.compute(ref_audio, est_audio)


if __name__ == '__main__':
    print("=== Speech BERT Score Demo ===")
    from revolab.metrics.base import create_demo_audio_data
    
    # This demo is slow as it downloads a large model on first run.
    print("Initializing metric (may download models)...")
    metric = SpeechBERT()
    
    demo_data = create_demo_audio_data()
    ref_audio = demo_data['clean']
    
    # The demo audio is a sine wave, so transcription will be nonsensical,
    # but it demonstrates the functionality.
    print("Computing score...")
    scores = metric.compute(ref_audio, demo_data['noisy'])
    
    print(f"Reference Transcription: '{scores['ref_transcription']}'")
    print(f"Estimated Transcription: '{scores['est_transcription']}'")
    print(f"BERT F1 Score: {scores['f1']:.4f}")