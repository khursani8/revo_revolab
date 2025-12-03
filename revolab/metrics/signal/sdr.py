"""
Signal to Distortion Ratio (SDR) metric implementation.

SDR measures the ratio between the desired signal and any unwanted distortion
in the estimated signal. It's widely used for evaluating source separation
and speech enhancement systems.
"""

import numpy as np
import librosa
from typing import Union, Optional

from ..base import AudioMetric, find_test_audio_files, demo_audio_files_exist, create_demo_audio_data


class SDR(AudioMetric):
    """
    Signal to Distortion Ratio metric.
    
    Computes the ratio between the desired signal and total distortion in the
    estimated signal. Higher SDR values indicate better quality.
    
    Typical SDR values (in dB):
    - > 20 dB: Excellent quality
    - 10-20 dB: Good quality
    - 0-10 dB: Fair quality
    - < 0 dB: Poor quality
    """
    
    def __init__(self, 
                 compute_permutation: bool = False,
                 sample_rate: int = 16000,
                 **kwargs):
        """
        Initialize SDR metric.
        
        Args:
            compute_permutation: Whether to compute optimal permutation for multichannel
            sample_rate: Audio sample rate
            **kwargs: Additional parameters
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.compute_permutation = compute_permutation
    
    def _compute_sdr(self, 
                   reference: np.ndarray, 
                   estimate: np.ndarray) -> float:
        """
        Compute SDR between reference and estimate.
        
        Args:
            reference: Reference signal
            estimate: Estimated signal
            
        Returns:
            SDR value in dB
        """
        # Ensure both signals have same length
        min_len = min(len(reference), len(estimate))
        reference = reference[:min_len]
        estimate = estimate[:min_len]
        
        # Compute energy of reference
        energy_ref = np.sum(reference ** 2)
        
        if energy_ref == 0:
            return float('-inf')
        
        # Compute optimal scaling factor
        alpha = np.dot(reference, estimate) / energy_ref
        
        # Scale the estimate
        scaled_estimate = alpha * estimate
        
        # Compute distortion
        distortion = reference - scaled_estimate
        
        # Compute energy of distortion
        energy_distortion = np.sum(distortion ** 2)
        
        if energy_distortion == 0:
            return float('inf')
        
        # Compute SDR in dB
        sdr = 10 * np.log10(energy_ref / energy_distortion)
        
        return float(sdr)
    
    def compute(self, 
               ref_audio: Union[str, np.ndarray], 
               est_audio: Union[str, np.ndarray]) -> float:
        """
        Compute SDR between reference and estimated audio.
        
        Args:
            ref_audio: Reference audio (file path or array)
            est_audio: Estimated audio (file path or array)
            
        Returns:
            SDR value in dB
        """
        # Load audio if paths are provided
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)
        
        # Preprocess audio
        ref_audio = self.preprocess_audio(ref_audio)
        est_audio = self.preprocess_audio(est_audio)
        
        return self._compute_sdr(ref_audio, est_audio)
    
    def compute_detailed(self, 
                        ref_audio: Union[str, np.ndarray], 
                        est_audio: Union[str, np.ndarray]) -> dict:
        """
        Compute detailed SDR analysis.
        
        Args:
            ref_audio: Reference audio (file path or array)
            est_audio: Estimated audio (file path or array)
            
        Returns:
            Dictionary with detailed SDR analysis
        """
        # Load audio if paths are provided
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)
        
        # Preprocess audio
        ref_audio = self.preprocess_audio(ref_audio)
        est_audio = self.preprocess_audio(est_audio)
        
        # Ensure same length
        min_len = min(len(ref_audio), len(est_audio))
        reference = ref_audio[:min_len]
        estimate = est_audio[:min_len]
        
        # Compute SDR
        sdr = self._compute_sdr(reference, estimate)
        
        # Compute additional statistics
        energy_ref = np.sum(reference ** 2)
        energy_est = np.sum(estimate ** 2)
        
        if energy_ref > 0:
            alpha = np.dot(reference, estimate) / energy_ref
            scaled_estimate = alpha * estimate
            distortion = reference - scaled_estimate
            energy_distortion = np.sum(distortion ** 2)
            
            # Compute correlation
            correlation = np.corrcoef(reference, estimate)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            alpha = 0.0
            energy_distortion = 0.0
            correlation = 0.0
        
        return {
            'sdr_db': sdr,
            'energy_reference': energy_ref,
            'energy_estimate': energy_est,
            'energy_distortion': energy_distortion,
            'scaling_factor': alpha,
            'correlation': correlation,
            'duration': len(reference) / self.sample_rate
        }
    
    def compute_frame_wise(self, 
                          ref_audio: Union[str, np.ndarray], 
                          est_audio: Union[str, np.ndarray],
                          frame_length: int = 1024,
                          hop_length: int = 512) -> np.ndarray:
        """
        Compute frame-wise SDR.
        
        Args:
            ref_audio: Reference audio (file path or array)
            est_audio: Estimated audio (file path or array)
            frame_length: Length of each frame
            hop_length: Hop length between frames
            
        Returns:
            Array of frame-wise SDR values
        """
        # Load audio if paths are provided
        if isinstance(ref_audio, str):
            ref_audio = self.load_audio(ref_audio)
        if isinstance(est_audio, str):
            est_audio = self.load_audio(est_audio)
        
        # Preprocess audio
        ref_audio = self.preprocess_audio(ref_audio)
        est_audio = self.preprocess_audio(est_audio)
        
        # Ensure same length
        min_len = min(len(ref_audio), len(est_audio))
        ref_audio = ref_audio[:min_len]
        est_audio = est_audio[:min_len]
        
        # Compute frame-wise SDR
        sdr_frames = []
        
        for i in range(0, len(ref_audio) - frame_length + 1, hop_length):
            ref_frame = ref_audio[i:i + frame_length]
            est_frame = est_audio[i:i + frame_length]
            
            frame_sdr = self._compute_sdr(ref_frame, est_frame)
            sdr_frames.append(frame_sdr)
        
        return np.array(sdr_frames)


def sdr(ref_audio: Union[str, np.ndarray], 
        est_audio: Union[str, np.ndarray],
        **kwargs) -> float:
    """
    Convenience function to compute SDR.
    
    Args:
        ref_audio: Reference audio (file path or array)
        est_audio: Estimated audio (file path or array)
        **kwargs: Additional parameters for SDR
        
    Returns:
        SDR value in dB
    """
    metric = SDR(**kwargs)
    return metric.compute(ref_audio, est_audio)


if __name__ == "__main__":
    """
    Demo script for SDR metric.
    
    Run this script to test SDR computation with demo audio or test files.
    """
    print("=== SDR (Signal to Distortion Ratio) Demo ===")
    
    # Initialize SDR metric
    sdr_metric = SDR()
    
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
                # Compute SDR
                sdr_value = sdr_metric.compute(ref_file, est_file)
                print(f"   SDR: {sdr_value:.2f} dB")
                
                # Get detailed analysis
                details = sdr_metric.compute_detailed(ref_file, est_file)
                print(f"   Reference energy: {details['energy_reference']:.2f}")
                print(f"   Estimate energy: {details['energy_estimate']:.2f}")
                print(f"   Distortion energy: {details['energy_distortion']:.2f}")
                print(f"   Correlation: {details['correlation']:.3f}")
                
                # Quality assessment
                if sdr_value > 20:
                    quality = "Excellent"
                elif sdr_value > 10:
                    quality = "Good"
                elif sdr_value > 0:
                    quality = "Fair"
                else:
                    quality = "Poor"
                print(f"   Quality assessment: {quality}")
                
            except Exception as e:
                print(f"   Error computing SDR: {e}")
        else:
            print("   Need at least 2 audio files for SDR comparison.")
    else:
        print("\n1. Testing with synthetic audio...")
        # Create demo audio
        demo_data = create_demo_audio_data()
        ref_audio = demo_data['clean']
        
        # Create distorted versions
        # 1. Add noise
        noisy_audio = ref_audio + 0.1 * np.random.randn(len(ref_audio))
        noisy_audio = noisy_audio.astype(np.float32)
        
        # 2. Scale and add noise
        scaled_audio = 0.8 * ref_audio + 0.2 * np.random.randn(len(ref_audio))
        scaled_audio = scaled_audio.astype(np.float32)
        
        # Test SDR
        print("   Testing different distortion levels:")
        
        # Clean vs noisy
        sdr_noisy = sdr_metric.compute(ref_audio, noisy_audio)
        print(f"   Clean vs Noisy (SNR~20dB): SDR = {sdr_noisy:.2f} dB")
        
        # Clean vs scaled
        sdr_scaled = sdr_metric.compute(ref_audio, scaled_audio)
        print(f"   Clean vs Scaled (0.8x): SDR = {sdr_scaled:.2f} dB")
        
        # Self-comparison (should be infinite)
        sdr_self = sdr_metric.compute(ref_audio, ref_audio)
        print(f"   Self-comparison (should be infinite): SDR = {sdr_self:.2f} dB")
        
        # Test with higher noise
        high_noise_audio = ref_audio + 0.3 * np.random.randn(len(ref_audio))
        high_noise_audio = high_noise_audio.astype(np.float32)
        sdr_high_noise = sdr_metric.compute(ref_audio, high_noise_audio)
        print(f"   Clean vs High Noise: SDR = {sdr_high_noise:.2f} dB")
    
    print("\n=== SDR Demo Complete ===")
    print("\nUsage examples:")
    print("1. With file paths:")
    print("   from revolab.metrics.signal import sdr")
    print("   score = sdr('reference.wav', 'estimated.wav')")
    print("\n2. With audio arrays:")
    print("   import librosa")
    print("   ref = librosa.load('reference.wav', sr=16000)[0]")
    print("   est = librosa.load('estimated.wav', sr=16000)[0]")
    print("   score = sdr(ref, est)")
