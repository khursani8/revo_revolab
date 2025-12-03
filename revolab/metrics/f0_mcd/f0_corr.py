"""
F0 Correlation metric implementation.

F0 correlation measures the similarity of fundamental frequency contours between
two audio signals. It's useful for evaluating pitch tracking accuracy and
prosody preservation in speech synthesis and voice conversion.
"""

import numpy as np
import librosa
from scipy.stats import pearsonr
from typing import Union, Tuple, Optional

from ..base import AudioMetric, find_test_audio_files, demo_audio_files_exist, create_demo_audio_data


class F0Correlation(AudioMetric):
    """
    F0 Correlation metric.
    
    Computes the Pearson correlation coefficient between F0 contours of 
    reference and estimated audio signals.
    
    Correlation values:
    - 1.0: Perfect correlation
    - 0.8-1.0: Excellent correlation
    - 0.6-0.8: Good correlation
    - 0.4-0.6: Fair correlation
    - 0.0-0.4: Poor correlation
    - < 0.0: Negative correlation (inverted pitch)
    """
    
    def __init__(self, 
                 fmin: float = 50.0,
                 fmax: float = 500.0,
                 hop_length: int = 256,
                 sample_rate: int = 16000,
                 **kwargs):
        """
        Initialize F0 Correlation metric.
        
        Args:
            fmin: Minimum F0 to track (Hz)
            fmax: Maximum F0 to track (Hz)
            hop_length: Hop length for pitch tracking
            sample_rate: Audio sample rate
            **kwargs: Additional parameters
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length = hop_length
    
    def _extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract F0 contour from audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            F0 contour array
        """
        # Use librosa's piptrack for F0 extraction
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fill_na=0.0
        )
        
        # Optional: Apply voicing threshold
        voicing_threshold = 0.1
        f0[voiced_probs < voicing_threshold] = 0.0
        
        return f0
    
    def _preprocess_f0(self, f0: np.ndarray) -> np.ndarray:
        """
        Preprocess F0 contour for correlation computation.
        
        Args:
            f0: Raw F0 contour
            
        Returns:
            Preprocessed F0 contour
        """
        # Keep only voiced frames (F0 > 0)
        voiced_f0 = f0[f0 > 0]
        
        return voiced_f0
    
    def compute(self, 
               ref_audio: Union[str, np.ndarray], 
               est_audio: Union[str, np.ndarray]) -> Tuple[float, int]:
        """
        Compute F0 correlation between reference and estimated audio.
        
        Args:
            ref_audio: Reference audio (file path or array)
            est_audio: Estimated audio (file path or array)
            
        Returns:
            Tuple of (correlation_coefficient, num_voiced_frames)
        """
        # Load audio if paths are provided
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)
        
        # Preprocess audio
        ref_audio = self.preprocess_audio(ref_audio)
        est_audio = self.preprocess_audio(est_audio)
        
        # Extract F0 contours
        ref_f0 = self._extract_f0(ref_audio)
        est_f0 = self._extract_f0(est_audio)
        
        # Ensure same length
        min_frames = min(len(ref_f0), len(est_f0))
        ref_f0 = ref_f0[:min_frames]
        est_f0 = est_f0[:min_frames]
        
        # Preprocess F0 (keep only voiced frames)
        ref_voiced = self._preprocess_f0(ref_f0)
        est_voiced = self._preprocess_f0(est_f0)
        
        # Need at least 2 voiced frames for correlation
        if len(ref_voiced) < 2 or len(est_voiced) < 2:
            return 0.0, max(len(ref_voiced), len(est_voiced))
        
        # Ensure same number of voiced frames
        min_voiced = min(len(ref_voiced), len(est_voiced))
        ref_voiced = ref_voiced[:min_voiced]
        est_voiced = est_voiced[:min_voiced]
        
        # Compute Pearson correlation
        try:
            correlation, _ = pearsonr(ref_voiced, est_voiced)
            # Handle NaN case
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        return float(correlation), len(ref_voiced)
    
    def compute_detailed(self, 
                        ref_audio: Union[str, np.ndarray], 
                        est_audio: Union[str, np.ndarray]) -> dict:
        """
        Compute detailed F0 analysis including raw contours and statistics.
        
        Args:
            ref_audio: Reference audio (file path or array)
            est_audio: Estimated audio (file path or array)
            
        Returns:
            Dictionary with detailed F0 analysis
        """
        # Load audio if paths are provided
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)
        
        # Preprocess audio
        ref_audio = self.preprocess_audio(ref_audio)
        est_audio = self.preprocess_audio(est_audio)
        
        # Extract F0 contours
        ref_f0 = self._extract_f0(ref_audio)
        est_f0 = self._extract_f0(est_audio)
        
        # Ensure same length
        min_frames = min(len(ref_f0), len(est_f0))
        ref_f0 = ref_f0[:min_frames]
        est_f0 = est_f0[:min_frames]
        
        # Compute basic statistics
        ref_voiced = ref_f0[ref_f0 > 0]
        est_voiced = est_f0[est_f0 > 0]
        
        correlation, voiced_frames = self.compute(ref_audio, est_audio)
        
        return {
            'correlation': correlation,
            'voiced_frames': voiced_frames,
            'total_frames': len(ref_f0),
            'voicing_ratio_ref': len(ref_voiced) / len(ref_f0) if len(ref_f0) > 0 else 0,
            'voicing_ratio_est': len(est_voiced) / len(est_f0) if len(est_f0) > 0 else 0,
            'f0_mean_ref': np.mean(ref_voiced) if len(ref_voiced) > 0 else 0,
            'f0_mean_est': np.mean(est_voiced) if len(est_voiced) > 0 else 0,
            'f0_std_ref': np.std(ref_voiced) if len(ref_voiced) > 0 else 0,
            'f0_std_est': np.std(est_voiced) if len(est_voiced) > 0 else 0,
            'ref_f0': ref_f0,
            'est_f0': est_f0
        }


def f0_corr(ref_audio: Union[str, np.ndarray], 
            est_audio: Union[str, np.ndarray],
            **kwargs) -> float:
    """
    Convenience function to compute F0 correlation.
    
    Args:
        ref_audio: Reference audio (file path or array)
        est_audio: Estimated audio (file path or array)
        **kwargs: Additional parameters for F0Correlation
        
    Returns:
        F0 correlation coefficient
    """
    metric = F0Correlation(**kwargs)
    correlation, _ = metric.compute(ref_audio, est_audio)
    return correlation


if __name__ == "__main__":
    """
    Demo script for F0 Correlation metric.
    
    Run this script to test F0 correlation computation with demo audio or test files.
    """
    print("=== F0 Correlation Demo ===")
    
    # Initialize F0 Correlation metric
    f0_metric = F0Correlation()
    
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
                # Compute F0 correlation
                correlation, voiced_frames = f0_metric.compute(ref_file, est_file)
                print(f"   F0 Correlation: {correlation:.3f}")
                print(f"   Voiced frames used: {voiced_frames}")
                
                # Get detailed analysis
                details = f0_metric.compute_detailed(ref_file, est_file)
                print(f"   Reference voicing ratio: {details['voicing_ratio_ref']:.3f}")
                print(f"   Estimated voicing ratio: {details['voicing_ratio_est']:.3f}")
                print(f"   Reference F0 mean: {details['f0_mean_ref']:.1f} Hz")
                print(f"   Estimated F0 mean: {details['f0_mean_est']:.1f} Hz")
                
                # Quality assessment
                if correlation >= 0.8:
                    quality = "Excellent"
                elif correlation >= 0.6:
                    quality = "Good"
                elif correlation >= 0.4:
                    quality = "Fair"
                elif correlation >= 0.0:
                    quality = "Poor"
                else:
                    quality = "Negative (inverted pitch)"
                print(f"   Quality assessment: {quality}")
                
            except Exception as e:
                print(f"   Error computing F0 correlation: {e}")
        else:
            print("   Need at least 2 audio files for F0 comparison.")
    else:
        print("\n1. Testing with synthetic audio...")
        # Create demo audio with different F0
        demo_data = create_demo_audio_data()
        
        # Modify the demo to have different F0
        sr = demo_data['sample_rate']
        t = np.linspace(0, 2.0, int(sr * 2.0), False)
        
        # Reference: 200 Hz fundamental
        ref_signal = (0.6 * np.sin(2 * np.pi * 200 * t) + 
                     0.3 * np.sin(2 * np.pi * 400 * t) +
                     0.1 * np.sin(2 * np.pi * 600 * t))
        ref_signal *= np.exp(-t * 0.5)
        ref_signal = ref_signal.astype(np.float32)
        
        # Estimated: 180 Hz fundamental (slightly different pitch)
        est_signal = (0.6 * np.sin(2 * np.pi * 180 * t) + 
                     0.3 * np.sin(2 * np.pi * 360 * t) +
                     0.1 * np.sin(2 * np.pi * 540 * t))
        est_signal *= np.exp(-t * 0.5)
        est_signal = est_signal.astype(np.float32)
        
        # Compute F0 correlation
        correlation, voiced_frames = f0_metric.compute(ref_signal, est_signal)
        print(f"   F0 Correlation between different pitch signals: {correlation:.3f}")
        print(f"   Voiced frames used: {voiced_frames}")
        
        # Test self-comparison (should be 1.0)
        corr_self, _ = f0_metric.compute(ref_signal, ref_signal)
        print(f"   Self-correlation (should be 1.0): {corr_self:.6f}")
    
    print("\n=== F0 Correlation Demo Complete ===")
    print("\nUsage examples:")
    print("1. With file paths:")
    print("   from revolab.metrics.f0_mcd import f0_corr")
    print("   score = f0_corr('reference.wav', 'estimated.wav')")
    print("\n2. With audio arrays:")
    print("   import librosa")
    print("   ref = librosa.load('reference.wav', sr=16000)[0]")
    print("   est = librosa.load('estimated.wav', sr=16000)[0]")
    print("   score = f0_corr(ref, est)")
