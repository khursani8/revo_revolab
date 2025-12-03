"""
Speech Token Distance metrics (WER, MER, CER).

This metric uses an Automatic Speech Recognition (ASR) model to transcribe
reference and estimated audio, then computes the distance between the
transcribed texts using various token distance metrics.
"""

from typing import Union, Dict, Any
from ..base import AudioMetric

try:
    from transformers import pipeline
    from jiwer import process_words, process_characters
except ImportError:
    pipeline = None
    process_words = None
    process_characters = None

class ASRTokenDistance(AudioMetric):
    """
    Speech Token Distance Metric.

    Transcribes audio and computes Word Error Rate (WER), Match Error Rate (MER),
    and Character Error Rate (CER).
    """
    def __init__(self, sample_rate: int = 16000, asr_model: str = "facebook/wav2vec2-base-960h", asr_pipeline=None, **kwargs):
        """
        Initialize SpeechTokenDistance metric.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if pipeline is None or process_words is None:
            raise ImportError("Please install torch, transformers, and jiwer to use this metric.")
        
        if asr_pipeline:
            self.asr_pipeline = asr_pipeline
        else:
            self.asr_pipeline = pipeline("automatic-speech-recognition", model=asr_model)

    def compute(self,
                ref_audio: Union[str, Any],
                est_audio: Union[str, Any]) -> Dict[str, Any]:
        """
        Compute Speech Token Distance metrics.
        """
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)

        ref_text = self.asr_pipeline(ref_audio)["text"]
        est_text = self.asr_pipeline(est_audio)["text"]

        word_results = process_words(ref_text, est_text)
        char_results = process_characters(ref_text, est_text)

        return {
            "wer": word_results.wer,
            "mer": word_results.mer,
            "cer": char_results.cer,
            "word_substitutions": word_results.substitutions,
            "word_deletions": word_results.deletions,
            "word_insertions": word_results.insertions,
            "ref_transcription": ref_text,
            "est_transcription": est_text,
        }

def speech_token_distance(ref_audio: Union[str, Any],
                          est_audio: Union[str, Any],
                          **kwargs) -> Dict[str, Any]:
    """
    Convenience function to compute Speech Token Distance metrics.
    """
    metric = SpeechTokenDistance(**kwargs)
    return metric.compute(ref_audio, est_audio)


if __name__ == '__main__':
    print("=== Speech Token Distance Demo ===")
    from revolab.metrics.base import create_demo_audio_data
    
    print("Initializing metric (may download models)...")
    metric = SpeechTokenDistance()
    
    demo_data = create_demo_audio_data()
    ref_audio = demo_data['clean']
    
    print("Computing score...")
    scores = metric.compute(ref_audio, demo_data['noisy'])
    
    print(f"Reference Transcription: '{scores['ref_transcription']}'")
    print(f"Estimated Transcription: '{scores['est_transcription']}'")
    print(f"WER: {scores['wer']:.4f}")
    print(f"CER: {scores['cer']:.4f}")
