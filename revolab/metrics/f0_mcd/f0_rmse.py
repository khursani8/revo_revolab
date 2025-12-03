"""
F0 Root Mean Square Error (RMSE) metric implementation.

F0 RMSE measures the average magnitude of the differences between 
fundamental frequency contours of reference and estimated audio signals.
It provides a direct measure of pitch deviation in Hz.
"""

import numpy as np
import librosa
from typing import Union, Tuple, Optional

from ..base import AudioMetric, find_test_audio_files, demo_audio_files_exist, create_demo_audio_data


class F0RMSE(AudioMetric):
    """
    F0 Root Mean Square Error metric.
    
    Computes the RMSE between F0 contours of reference and estimated audio signals.
    Lower RMSE values indicate better pitch accuracy.
    
    Typical RMSE values (in Hz):
    - < 10 Hz: Excellent pitch accuracy
    - 10-20 Hz: Good pitch accuracy
    - 20-40 Hz: Fair pitch accuracy
    - > 40 Hz: Poor pitch accuracy
    """
    
    def __init__(self, 
                 fmin: float = 50.0,
                 fmax: float = 500.0,
                 hop_length: int = 256,
                 sample_rate: int = 16000,
                 **kwargs):
        """
        Initialize F0 RMSE metric.
        
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
        
        # Apply voicing threshold
        voicing_threshold = 0.1
        f0[voiced_probs < voicing_threshold] = 0.0
        
        return f0
    
    def _compute_frame_wise_rmse(self, 
                                 ref_f0: np.ndarray, 
                                 est_f0: np.ndarray) -> np.ndarray:
        """
        Compute frame-wise RMSE between F0 contours.
        
        Args:
            ref_f0: Reference F0 contour
            est_f0: Estimated F0 contour
            
        Returns:
            Frame-wise RMSE array
        """
        # Find frames where both are voiced (F0 > 0)
        both_voiced = (ref_f0 > 0) & (est_f0 > 0)
        
        if not np.any(both_voiced):
            return np.array([0.0])
        
        # Compute squared differences for voiced frames
        diff_squared = (ref_f0[both_voiced] - est_f0[both_voiced]) ** 2
        
        # Compute RMSE
        rmse = np.sqrt(np.mean(diff_squared))
        
        return np.array([rmse])
    
    def compute(self, 
               ref_audio: Union[str, np.ndarray], 
               est_audio: Union[str, np.ndarray]) -> Tuple[float, int]:
        """
        Compute F0 RMSE between reference and estimated audio.
        
        Args:
            ref_audio: Reference audio (file path or array)
            est_audio: Estimated audio (file path or array)
            
        Returns:
            Tuple of (rmse_value_hz, num_voiced_frames_used)
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
        
        # Find frames where both are voiced
        both_voiced = (ref_f0 > 0) & (est_f0 > 0)
        voiced_frames = np.sum(both_voiced)
        
        if voiced_frames == 0:
            return float('inf'), 0
        
        # Compute RMSE for voiced frames
        ref_voiced = ref_f0[both_voiced]
        est_voiced = est_f0[both_voiced]
        
        mse = np.mean((ref_voiced - est_voiced) ** 2)
        rmse = np.sqrt(mse)
        
        return float(rmse), int(voiced_frames)
    
    def compute_detailed(self, 
                        ref_audio: Union[str, np.ndarray], 
                        est_audio: Union[str, np.ndarray]) -> dict:
        """
        Compute detailed F0 RMSE analysis.
        
        Args:
            ref_audio: Reference audio (file path or array)
            est_audio: Estimated audio (file path or array)
            
        Returns:
            Dictionary with detailed F0 RMSE analysis
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
        
        # Find frames where both are voiced
        both_voiced = (ref_f0 > 0) & (est_f0 > 0)
        voiced_frames = np.sum(both_voiced)
        
        rmse, _ = self.compute(ref_audio, est_audio)
        
        # Additional statistics
        if voiced_frames > 0:
            ref_both = ref_f0[both_voiced]
            est_both = est_f0[both_voiced]
            
            mean_error = np.mean(ref_both - est_both)
            std_error = np.std(ref_both - est_both)
            max_error = np.max(np.abs(ref_both - est_both))
            
            # Percentage error
            ref_mean = np.mean(ref_both)
            percentage_error = (rmse / ref_mean * 100) if ref_mean > 0 else float('inf')
        else:
            mean_error = 0.0
            std_error = 0.0
            max_error = 0.0
            percentage_error = float('inf')
        
        return {
            'rmse_hz': rmse,
            'voiced_frames_used': voiced_frames,
            'total_frames': len(ref_f0),
            'mean_error_hz': mean_error,
            'std_error_hz': std_error,
            'max_error_hz': max_error,
            'percentage_error': percentage_error,
            'voicing_ratio_ref': len(ref_voiced) / len(ref_f0) if len(ref_f0) > 0 else 0,
            'voicing_ratio_est': len(est_voiced) / len(est_f0) if len(est_f0) > 0 else 0,
            'f0_mean_ref': np.mean(ref_voiced) if len(ref_voiced) > 0 else 0,
            'f0_mean_est': np.mean(est_voiced) if len(est_voiced) > 0 else 0,
            'ref_f0': ref_f0,
            'est_f0': est_f0
        }
    
    def compute_frame_wise(self, 
                          ref_audio: Union[str, np.ndarray], 
                          est_audio: Union[str, np.ndarray]) -> np.ndarray:
        """
        Compute frame-wise F0 errors.
        
        Args:
            ref_audio: Reference audio (file path or array)
            est_audio: Estimated audio (file path or array)
            
        Returns:
            Array of frame-wise F0 errors (Hz)
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
        
        # Compute frame-wise errors
        frame_errors = np.abs(ref_f0 - est_f0)
        
        # Set errors to 0 where either is unvoiced
        both_voiced = (ref_f0 > 0) & (est_f0 > 0)
        frame_errors[~both_voiced] = 0.0
        
        return frame_errors


def f0_rmse(ref_audio: Union[str, np.ndarray], 
             est_audio: Union[str, np.ndarray],
             **kwargs) -> float:
    """
    Convenience function to compute F0 RMSE.
    
    Args:
        ref_audio: Reference audio (file path or array)
        est_audio: Estimated audio (file path or array)
        **kwargs: Additional parameters for F0RMSE
        
    Returns:
        F0 RMSE in Hz
    """
    metric = F0RMSE(**kwargs)
    rmse, _ = metric.compute(ref_audio, est_audio)
    return rmse


if __name__ == "__main__":
    """
    Demo script for F0 RMSE metric.
    
    Run this script to test F0 RMSE computation with demo audio or test files.
    """
    print("=== F0 RMSE Demo ===")
    
    # Initialize F0 RMSE metric
    f0_rmse_metric = F0RMSE()
    
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
                # Compute F0 RMSE
                rmse, voiced_frames = f0_rmse_metric.compute(ref_file, est_file)
                print(f"   F0 RMSE: {rmse:.2f} Hz")
                print(f"   Voiced frames used: {voiced_frames}")
                
                # Get detailed analysis
                details = f0_rmse_metric.compute_detailed(ref_file, est_file)
                print(f"   Mean error: {details['mean_error_hz']:.2f} Hz")
                print(f"   Max error: {details['max_error_hz']:.2f} Hz")
                print(f"   Percentage error: {details['percentage_error']:.1f}%")
                
                # Quality assessment
                if rmse < 10:
                    quality = "Excellent"
                elif rmse < 20:
                    quality = "Good"
                elif rmse < 40:
                    quality = "Fair"
                else:
                    quality = "Poor"
                print(f"   Quality assessment: {quality}")
                
            except Exception as e:
                print(f"   Error computing F0 RMSE: {e}")
        else:
            print("   Need at least 2 audio files for F0 RMSE comparison.")
    else:
        print("\n1. Testing with synthetic audio...")
        # Create demo audio with known F0 difference
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
        
        # Estimated: 180 Hz fundamental (20 Hz difference)
        est_signal = (0.6 * np.sin(2 * np.pi * 180 * t) + 
                     0.3 * np.sin(2 * np.pi * 360 * t) +
                     0.1 * np.sin(2 * np.pi * 540 * t))
        est_signal *= np.exp(-t * 0.5)
        est_signal = est_signal.astype(np.float32)
        
        # Compute F0 RMSE
        rmse, voiced_frames = f0_rmse_metric.compute(ref_signal, est_signal)
        print(f"   F0 RMSE between 200Hz and 180Hz signals: {rmse:.2f} Hz")
        print(f"   Expected RMSE: ~20 Hz (difference between fundamentals)")
        print(f"   Voiced frames used: {voiced_frames}")
        
        # Test self-comparison (should be 0)
        rmse_self, _ = f0_rmse_metric.compute(ref_signal, ref_signal)
        print(f"   Self-RMSE (should be 0): {rmse_self:.6f} Hz")
        
        # Test with larger difference
        est_signal_250 = (0.6 * np.sin(2 * np.pi * 250 * t) + 
                          0.3 * np.sin(2 * np.pi * 500 * t) +
                          0.1 * np.sin(2 * np.pi * 750 * t))
        est_signal_250 *= np.exp(-t * 0.5)
        est_signal_250 = est_signal_250.astype(np.float32)
        
        rmse_250, _ = f0_rmse_metric.compute(ref_signal, est_signal_250)
        print(f"   F0 RMSE between 200Hz and 250Hz signals: {rmse_250:.2f} Hz")
        print(f"   Expected RMSE: ~50 Hz (difference between fundamentals)")
    
    print("\n=== F0 RMSE Demo Complete ===")
    print("\nUsage examples:")
    print("1. With file paths:")
    print("   from revolab.metrics.f0_mcd import f0_rmse")
    print("   score = f0_rmse('reference.wav', 'estimated.wav')")
    print("\n2. With audio arrays:")
    print("   import librosa")
    print("   ref = librosa.load('reference.wav', sr=16000)[0]")
    print("   est = librosa.load('estimated.wav', sr=16000)[0]")
    print("   score = f0_rmse(ref, est)")
