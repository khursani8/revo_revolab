"""
Unit tests for the SI-SDR metric.
"""
import unittest
import numpy as np
from revolab.metrics.signal.si_sdr import SISDR, si_sdr
from revolab.metrics.base import create_demo_audio_data

class TestSISDR(unittest.TestCase):
    def setUp(self):
        self.metric = SISDR()
        demo_data = create_demo_audio_data()
        self.ref_audio = demo_data['clean']
        self.est_audio_scaled = 0.5 * self.ref_audio
        self.est_audio_noisy = (self.ref_audio + 0.1 * np.random.randn(len(self.ref_audio))).astype(np.float32)

    def test_si_sdr_class(self):
        """Test SISDR class compute method."""
        # Test with scaled audio (should be infinite)
        sdr_val = self.metric.compute(self.ref_audio, self.est_audio_scaled)
        self.assertEqual(sdr_val, np.inf)

        # Test with noisy audio
        sdr_val = self.metric.compute(self.ref_audio, self.est_audio_noisy)
        self.assertIsInstance(sdr_val, float)
        self.assertGreater(sdr_val, 0)

    def test_si_sdr_function(self):
        """Test si_sdr convenience function."""
        # Test with scaled audio (should be infinite)
        sdr_val = si_sdr(self.ref_audio, self.est_audio_scaled)
        self.assertEqual(sdr_val, np.inf)

        # Test with noisy audio
        sdr_val = si_sdr(self.ref_audio, self.est_audio_noisy)
        self.assertIsInstance(sdr_val, float)
        self.assertGreater(sdr_val, 0)
        
    def test_zero_signal(self):
        """Test with zero reference signal."""
        zero_ref = np.zeros_like(self.ref_audio)
        sdr_val = si_sdr(zero_ref, self.est_audio_noisy)
        self.assertEqual(sdr_val, -np.inf)

if __name__ == '__main__':
    unittest.main()
