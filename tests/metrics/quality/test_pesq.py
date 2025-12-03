"""
Unit tests for the PESQ metric.
"""
import unittest
import numpy as np
from revolab.metrics.quality.pesq import PESQ, pesq
from revolab.metrics.base import create_demo_audio_data

class TestPESQ(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.metric = PESQ(sample_rate=self.sample_rate)
        demo_data = create_demo_audio_data()
        self.ref_audio = demo_data['clean']
        self.deg_audio = demo_data['noisy']

    def test_pesq_class(self):
        """Test PESQ class compute method."""
        score = self.metric.compute(self.ref_audio, self.deg_audio)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -0.5)
        self.assertLessEqual(score, 4.5)

    def test_pesq_function(self):
        """Test pesq convenience function."""
        score = pesq(self.ref_audio, self.deg_audio, sample_rate=self.sample_rate)
        self.assertIsInstance(score, float)

    def test_invalid_sample_rate(self):
        """Test with an invalid sample rate."""
        with self.assertRaises(ValueError):
            PESQ(sample_rate=22050)

    def test_invalid_mode(self):
        """Test with an invalid mode."""
        with self.assertRaises(ValueError):
            PESQ(sample_rate=16000, mode='invalid')

if __name__ == '__main__':
    unittest.main()
