"""
Speech BLEU Score metric.

This metric uses an Automatic Speech Recognition (ASR) model to transcribe
reference and estimated audio, then computes the BLEU score between the
transcribed texts.
"""

from typing import Union, Dict, Any
from ..base import AudioMetric

try:
    from transformers import pipeline
    from sacrebleu import corpus_bleu
except ImportError:
    pipeline = None
    corpus_bleu = None

class SpeechBLEU(AudioMetric):
    """
    Speech BLEU Score Metric.

    Transcribes reference and estimated audio using an ASR model and then
    computes the BLEU score between the transcriptions.
    """
    def __init__(self, sample_rate: int = 16000, asr_model: str = "facebook/wav2vec2-base-960h", asr_pipeline=None, **kwargs):
        """
        Initialize SpeechBLEU metric.

        Args:
            sample_rate: Audio sample rate.
            asr_model: The ASR model to use.
            asr_pipeline: An optional pre-initialized ASR pipeline.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if pipeline is None or corpus_bleu is None:
            raise ImportError("Please install torch, transformers, and sacrebleu to use the SpeechBLEU metric.")
        
        if asr_pipeline:
            self.asr_pipeline = asr_pipeline
        else:
            self.asr_pipeline = pipeline("automatic-speech-recognition", model=asr_model)

    def compute(self,
                ref_audio: Union[str, Any],
                est_audio: Union[str, Any]) -> Dict[str, Any]:
        """
        Compute the Speech BLEU score.

        Args:
            ref_audio: The reference audio.
            est_audio: The estimated audio.

        Returns:
            A dictionary containing the BLEU score and transcriptions.
        """
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)

        ref_text = self.asr_pipeline(ref_audio)["text"]
        est_text = self.asr_pipeline(est_audio)["text"]

        bleu = corpus_bleu([est_text], [[ref_text]])

        return {
            "score": bleu.score,
            "counts": bleu.counts,
            "totals": bleu.totals,
            "precisions": bleu.precisions,
            "bp": bleu.bp,
            "sys_len": bleu.sys_len,
            "ref_len": bleu.ref_len,
            "ref_transcription": ref_text,
            "est_transcription": est_text,
        }

def speech_bleu(ref_audio: Union[str, Any],
                est_audio: Union[str, Any],
                **kwargs) -> Dict[str, Any]:
    """
    Convenience function to compute Speech BLEU score.
    """
    metric = SpeechBLEU(**kwargs)
    return metric.compute(ref_audio, est_audio)


if __name__ == '__main__':
    print("=== Speech BLEU Score Demo ===")
    from revolab.metrics.base import create_demo_audio_data
    
    print("Initializing metric (may download models)...")
    metric = SpeechBLEU()
    
    demo_data = create_demo_audio_data()
    ref_audio = demo_data['clean']
    
    print("Computing score...")
    scores = metric.compute(ref_audio, demo_data['noisy'])
    
    print(f"Reference Transcription: '{scores['ref_transcription']}'")
    print(f"Estimated Transcription: '{scores['est_transcription']}'")
    print(f"BLEU Score: {scores['score']:.2f}")
