"""
Mel Cepstral Distortion (MCD) metric implementation.

MCD measures the spectral distance between two audio signals in the mel-cepstral domain.
It's commonly used to evaluate speech synthesis and voice conversion systems.
"""

import numpy as np
import librosa
from typing import Union, Optional

from ..base import AudioMetric, find_test_audio_files, demo_audio_files_exist, create_demo_audio_data


class MCD(AudioMetric):
    """
    Mel Cepstral Distortion metric.
    
    Computes the distance between mel-cepstral coefficients of reference and estimated signals.
    Lower MCD values indicate better quality (closer to reference).
    
    Typical MCD values:
    - < 1.0: Excellent quality
    - 1.0-2.5: Good quality
    - 2.5-4.0: Fair quality
    - > 4.0: Poor quality
    """
    
    def __init__(self, 
                 n_mfcc: int = 13, 
                 n_fft: int = 1024, 
                 hop_length: int = 256,
                 n_mels: int = 80,
                 sample_rate: int = 16000,
                 **kwargs):
        """
        Initialize MCD metric.
        
        Args:
            n_mfcc: Number of MFCC coefficients to compute
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filter banks
            sample_rate: Audio sample rate
            **kwargs: Additional parameters
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def _extract_mel_cepstrum(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-cepstral coefficients from audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Mel-cepstral coefficients matrix
        """
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        return mfcc
    
    def compute(self, 
               ref_audio: Union[str, np.ndarray], 
               est_audio: Union[str, np.ndarray]) -> float:
        """
        Compute MCD between reference and estimated audio.
        
        Args:
            ref_audio: Reference audio (file path or array)
            est_audio: Estimated audio (file path or array)
            
        Returns:
            MCD value in dB
        """
        # Load audio if paths are provided
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)
        
        # Preprocess audio
        ref_audio = self.preprocess_audio(ref_audio)
        est_audio = self.preprocess_audio(est_audio)
        
        # Extract mel-cepstral coefficients
        ref_mfcc = self._extract_mel_cepstrum(ref_audio)
        est_mfcc = self._extract_mel_cepstrum(est_audio)
        
        # Ensure same length by truncating to shortest
        min_frames = min(ref_mfcc.shape[1], est_mfcc.shape[1])
        ref_mfcc = ref_mfcc[:, :min_frames]
        est_mfcc = est_mfcc[:, :min_frames]
        
        # Compute frame-wise MCD
        # MCD formula: sqrt(2) * sqrt(sum((c1 - c2)^2) - c1[0]*c2[0])
        diff = ref_mfcc - est_mfcc
        
        # Remove 0th coefficient (energy) for MCD calculation
        mcd_frames = np.sqrt(2.0) * np.sqrt(np.sum(diff[1:, :] ** 2, axis=0))
        
        # Return mean MCD across all frames
        return float(np.mean(mcd_frames))
    
    def compute_frame_wise(self, 
                          ref_audio: Union[str, np.ndarray], 
                          est_audio: Union[str, np.ndarray]) -> np.ndarray:
        """
        Compute frame-wise MCD.
        
        Args:
            ref_audio: Reference audio (file path or array)
            est_audio: Estimated audio (file path or array)
            
        Returns:
            Array of frame-wise MCD values
        """
        # Load audio if paths are provided
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)
        
        # Preprocess audio
        ref_audio = self.preprocess_audio(ref_audio)
        est_audio = self.preprocess_audio(est_audio)
        
        # Extract mel-cepstral coefficients
        ref_mfcc = self._extract_mel_cepstrum(ref_audio)
        est_mfcc = self._extract_mel_cepstrum(est_audio)
        
        # Ensure same length
        min_frames = min(ref_mfcc.shape[1], est_mfcc.shape[1])
        ref_mfcc = ref_mfcc[:, :min_frames]
        est_mfcc = est_mfcc[:, :min_frames]
        
        # Compute frame-wise MCD
        diff = ref_mfcc - est_mfcc
        mcd_frames = np.sqrt(2.0) * np.sqrt(np.sum(diff[1:, :] ** 2, axis=0))
        
        return mcd_frames


def mcd(ref_audio: Union[str, np.ndarray], 
        est_audio: Union[str, np.ndarray],
        **kwargs) -> float:
    """
    Convenience function to compute MCD.
    
    Args:
        ref_audio: Reference audio (file path or array)
        est_audio: Estimated audio (file path or array)
        **kwargs: Additional parameters for MCD
        
    Returns:
        MCD value in dB
    """
    metric = MCD(**kwargs)
    return metric.compute(ref_audio, est_audio)


if __name__ == "__main__":
    """
    Demo script for MCD metric.
    
    Run this script to test MCD computation with demo audio or test files.
    """
    print("=== MCD (Mel Cepstral Distortion) Demo ===")
    
    # Initialize MCD metric
    mcd_metric = MCD()
    
    if demo_audio_files_exist():
        print("\n1. Testing with real audio files...")
        audio_files = find_test_audio_files()
        
        if len(audio_files) >= 2:
            # Use first two audio files
            ref_file = audio_files[0]
            est_file = audio_files[1]
            
            print(f"   Reference: {ref_file}")
            print(f"   Estimated: {est_file}")
            
            try:
                # Compute MCD
                mcd_value = mcd_metric.compute(ref_file, est_file)
                print(f"   MCD: {mcd_value:.3f} dB")
                
                # Compute frame-wise MCD
                mcd_frames = mcd_metric.compute_frame_wise(ref_file, est_file)
                print(f"   Frame-wise MCD: mean={np.mean(mcd_frames):.3f}, "
                      f"std={np.std(mcd_frames):.3f}")
                print(f"   Number of frames: {len(mcd_frames)}")
                
                # Quality assessment
                if mcd_value < 1.0:
                    quality = "Excellent"
                elif mcd_value < 2.5:
                    quality = "Good"
                elif mcd_value < 4.0:
                    quality = "Fair"
                else:
                    quality = "Poor"
                print(f"   Quality assessment: {quality}")
                
            except Exception as e:
                print(f"   Error computing MCD: {e}")
        else:
            print("   Need at least 2 audio files for MCD comparison.")
    else:
        print("\n1. Testing with synthetic audio...")
        # Create demo audio
        demo_data = create_demo_audio_data()
        ref_audio = demo_data['clean']
        est_audio = demo_data['noisy']
        
        # Compute MCD
        mcd_value = mcd_metric.compute(ref_audio, est_audio)
        print(f"   MCD between clean and noisy synthetic audio: {mcd_value:.3f} dB")
        
        # Test self-comparison (should be ~0)
        mcd_self = mcd_metric.compute(ref_audio, ref_audio)
        print(f"   MCD self-comparison (should be ~0): {mcd_self:.6f} dB")
    
    print("\n=== MCD Demo Complete ===")
    print("\nUsage examples:")
    print("1. With file paths:")
    print("   from revolab.metrics.f0_mcd import mcd")
    print("   score = mcd('reference.wav', 'estimated.wav')")
    print("\n2. With audio arrays:")
    print("   import librosa")
    print("   ref = librosa.load('reference.wav', sr=16000)[0]")
    print("   est = librosa.load('estimated.wav', sr=16000)[0]")
    print("   score = mcd(ref, est)")
