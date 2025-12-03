"""
Unit tests for the SIR metric.
"""
import unittest
import numpy as np
from revolab.metrics.signal.sir import SIR, sir
from revolab.metrics.base import create_demo_audio_data

class TestSIR(unittest.TestCase):
    def setUp(self):
        self.metric = SIR()
        demo_data = create_demo_audio_data()
        self.ref_audio = demo_data['clean']
        
        # Create a simple interference signal
        sr = demo_data['sample_rate']
        t = np.linspace(0, len(self.ref_audio) / sr, len(self.ref_audio), False)
        self.interference = 0.5 * np.sin(2 * np.pi * 500 * t)
        self.interference = self.interference.astype(np.float32)

    def test_sir_class(self):
        """Test SIR class compute method."""
        sir_val = self.metric.compute(self.ref_audio, self.interference)
        self.assertIsInstance(sir_val, float)

    def test_sir_function(self):
        """Test sir convenience function."""
        sir_val = sir(self.ref_audio, self.interference)
        self.assertIsInstance(sir_val, float)
        
    def test_zero_interference(self):
        """Test with zero interference signal."""
        zero_interference = np.zeros_like(self.ref_audio)
        sir_val = sir(self.ref_audio, zero_interference)
        self.assertEqual(sir_val, np.inf)

if __name__ == '__main__':
    unittest.main()
