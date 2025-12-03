"""
Unit tests for the SI-SNRI metric.
"""
import unittest
import numpy as np
from revolab.metrics.signal.si_snri import SISNRI, si_snri
from revolab.metrics.base import create_demo_audio_data

class TestSISNRI(unittest.TestCase):
    def setUp(self):
        self.metric = SISNRI()
        demo_data = create_demo_audio_data()
        self.clean_audio = demo_data['clean']
        self.noisy_audio = demo_data['noisy']
        self.enhanced_audio = (self.clean_audio + 0.05 * np.random.randn(len(self.clean_audio))).astype(np.float32)

    def test_si_snri_class(self):
        """Test SISNRI class compute method."""
        improvement = self.metric.compute(self.clean_audio, self.noisy_audio, self.enhanced_audio)
        self.assertIsInstance(improvement, float)
        self.assertGreater(improvement, 0)

    def test_si_snri_function(self):
        """Test si_snri convenience function."""
        improvement = si_snri(self.clean_audio, self.noisy_audio, self.enhanced_audio)
        self.assertIsInstance(improvement, float)
        self.assertGreater(improvement, 0)
        
    def test_negative_improvement(self):
        """Test with an enhanced signal that is worse than the noisy signal."""
        worse_audio = (self.clean_audio + 0.2 * np.random.randn(len(self.clean_audio))).astype(np.float32)
        improvement = si_snri(self.clean_audio, self.noisy_audio, worse_audio)
        self.assertLess(improvement, 0)

if __name__ == '__main__':
    unittest.main()
