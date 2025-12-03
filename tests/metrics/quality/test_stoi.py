"""
Unit tests for the STOI metric.
"""
import unittest
import numpy as np
from revolab.metrics.quality.stoi import STOI, stoi
from revolab.metrics.base import create_demo_audio_data

class TestSTOI(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.metric = STOI(sample_rate=self.sample_rate)
        demo_data = create_demo_audio_data()
        self.ref_audio = demo_data['clean']
        self.deg_audio = demo_data['noisy']

    def test_stoi_class(self):
        """Test STOI class compute method."""
        score = self.metric.compute(self.ref_audio, self.deg_audio)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_stoi_function(self):
        """Test stoi convenience function."""
        score = stoi(self.ref_audio, self.deg_audio, sample_rate=self.sample_rate)
        self.assertIsInstance(score, float)

    def test_invalid_sample_rate(self):
        """Test with an invalid sample rate."""
        with self.assertRaises(ValueError):
            STOI(sample_rate=8000)

if __name__ == '__main__':
    unittest.main()
