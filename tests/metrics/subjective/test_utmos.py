"""
Unit tests for the UT-MOS metric.
"""
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

class TestUTMOS(unittest.TestCase):

    def setUp(self):
        """Set up the test case."""
        self.ref_audio = np.random.randn(16000)

    @patch('torch.hub.load')
    def test_utmos_with_mock_model(self, mock_torch_hub_load):
        """Test the UTMOS metric using a mocked model."""
        # Configure the mock predictor
        mock_predictor = MagicMock()
        # The predictor should return a tensor-like object with an item() method
        mock_predictor.return_value = MagicMock(item=lambda: 4.5)
        
        # Configure torch.hub.load to return our mock predictor
        mock_torch_hub_load.return_value = mock_predictor

        # We must import the module *after* the patch is in place for it to take effect
        from revolab.metrics.subjective.utmos import UTMOS
        
        # Initialize the metric, which will now use the mocked model
        metric = UTMOS()
        
        # Run the computation
        score = metric.compute(self.ref_audio)

        # Assertions
        self.assertIsInstance(score, float)
        self.assertEqual(score, 4.5)
        
        # Verify that torch.hub.load was called correctly
        mock_torch_hub_load.assert_called_with(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True
        )
        
        # Verify that the predictor was called
        self.assertTrue(mock_predictor.called)

if __name__ == '__main__':
    unittest.main()
