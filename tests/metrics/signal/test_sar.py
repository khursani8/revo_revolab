"""
Unit tests for the SAR metric.
"""
import unittest
import numpy as np
from revolab.metrics.signal.sar import SAR, sar
from revolab.metrics.base import create_demo_audio_data

class TestSAR(unittest.TestCase):
    def setUp(self):
        self.metric = SAR()
        demo_data = create_demo_audio_data()
        self.ref_audio = demo_data['clean']
        self.est_audio = demo_data['noisy']

    def test_sar_class(self):
        """Test SAR class compute method."""
        sar_val = self.metric.compute(self.ref_audio, self.est_audio)
        self.assertIsInstance(sar_val, float)

    def test_sar_function(self):
        """Test sar convenience function."""
        sar_val = sar(self.ref_audio, self.est_audio)
        self.assertIsInstance(sar_val, float)
        
    def test_perfect_signal(self):
        """Test with a perfect signal (no artifacts)."""
        sar_val = sar(self.ref_audio, self.ref_audio)
        self.assertEqual(sar_val, np.inf)

if __name__ == '__main__':
    unittest.main()
