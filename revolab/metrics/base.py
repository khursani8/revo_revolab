"""
Base classes and utilities for speech metrics.

This module provides the foundation for all speech metrics in revolab,
including base classes, common utilities, and helper functions.
"""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import librosa


class BaseMetric(ABC):
    """
    Abstract base class for all speech metrics.
    
    All metrics should inherit from this class and implement the required methods.
    """
    
    def __init__(self, sample_rate: int = 16000, **kwargs):
        """
        Initialize the metric.
        
        Args:
            sample_rate: Target sample rate for audio processing
            **kwargs: Additional metric-specific parameters
        """
        self.sample_rate = sample_rate
        self.kwargs = kwargs
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> Union[float, np.ndarray, Dict[str, Any]]:
        """
        Compute the metric.
        
        This method must be implemented by each metric.
        
        Returns:
            Metric value(s)
        """
        pass
    
    def __call__(self, *args, **kwargs) -> Union[float, np.ndarray, Dict[str, Any]]:
        """Make the metric callable."""
        return self.compute(*args, **kwargs)


class AudioMetric(BaseMetric):
    """
    Base class for metrics that operate on audio signals.
    
    Provides common audio loading and preprocessing functionality.
    """
    
    def load_audio(self, audio_path: str, sr: Optional[int] = None) -> np.ndarray:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            sr: Target sample rate (uses self.sample_rate if None)
            
        Returns:
            Audio array as float32 normalized to [-1, 1]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        sr = sr or self.sample_rate
        try:
            audio, loaded_sr = librosa.load(audio_path, sr=sr, mono=True)
            return audio.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio {audio_path}: {e}")
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for metric computation.
        
        Args:
            audio: Input audio array
            
        Returns:
            Preprocessed audio array
        """
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        return audio


def find_test_audio_files(directory: str = "test_data", pattern: str = "*.wav") -> List[str]:
    """
    Find test audio files in the specified directory.
    
    Args:
        directory: Directory to search for audio files
        pattern: File pattern to match
        
    Returns:
        List of audio file paths
    """
    import glob
    
    if not os.path.exists(directory):
        warnings.warn(f"Test data directory not found: {directory}")
        return []
    
    files = glob.glob(os.path.join(directory, pattern))
    return sorted(files)


def create_demo_audio_data():
    """
    Create synthetic demo audio data for testing purposes.
    
    Returns:
        Dictionary with demo audio signals
    """
    sr = 16000
    duration = 2.0  # seconds
    
    # Create a simple sine wave speech-like signal
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # Fundamental frequency around 200 Hz (typical male voice)
    f0 = 200
    
    # Add harmonics for speech-like quality
    signal = (
        0.6 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
        0.2 * np.sin(2 * np.pi * 3 * f0 * t) +
        0.1 * np.sin(2 * np.pi * 4 * f0 * t)
    )
    
    # Add some envelope to make it more speech-like
    envelope = np.exp(-t * 0.5) * (1 - np.exp(-t * 10))
    signal *= envelope
    
    return {
        'clean': signal.astype(np.float32),
        'noisy': (signal + 0.1 * np.random.randn(len(signal))).astype(np.float32),
        'sample_rate': sr
    }


def demo_audio_files_exist(directory: str = "test_data") -> bool:
    """
    Check if demo audio files exist in the test directory.
    
    Args:
        directory: Directory to check
        
    Returns:
        True if audio files exist, False otherwise
    """
    files = find_test_audio_files(directory)
    return len(files) > 0


if __name__ == "__main__":
    """
    Demo script for base metric functionality.
    
    Run this script to test the base classes and utilities.
    """
    print("=== Revolab Base Metric Demo ===")
    
    # Test demo audio creation
    print("\n1. Creating demo audio data...")
    demo_data = create_demo_audio_data()
    print(f"   Created demo audio: {len(demo_data['clean'])} samples at {demo_data['sample_rate']}Hz")
    
    # Test audio file detection
    print("\n2. Checking for test audio files...")
    if demo_audio_files_exist():
        files = find_test_audio_files()
        print(f"   Found {len(files)} test audio files:")
        for f in files:
            print(f"     - {f}")
    else:
        print("   No test audio files found. Add WAV files to test_data/ directory.")
        print("   See test_data/README.md for format requirements.")
    
    # Test audio loading (if files exist)
    if demo_audio_files_exist():
        print("\n3. Testing audio loading...")
        files = find_test_audio_files()[:2]  # Test first 2 files
        for file_path in files:
            try:
                metric = AudioMetric()
                audio = metric.load_audio(file_path)
                print(f"   Loaded {file_path}: {len(audio)} samples, "
                      f"duration={len(audio)/16000:.2f}s")
            except Exception as e:
                print(f"   Failed to load {file_path}: {e}")
    
    print("\n=== Base Metric Demo Complete ===")
    print("\nNext steps:")
    print("1. Add audio files to test_data/ directory")
    print("2. Run individual metric demos:")
    print("   uv run python -m revolab.metrics.f0_mcd.mcd")
    print("   uv run python -m revolab.metrics.signal.sdr")
    print("   uv run python -m revolab.metrics.quality.pesq")
