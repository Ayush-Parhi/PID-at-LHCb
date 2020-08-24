"""Tests for CNNmodel class."""
import os
from pathlib import Path
import unittest

from cnn_predictor import CnnPredictor

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "support"

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestCnnPredictor(unittest.TestCase):
    """Tests for the CnnPredictor class."""

    def test_filename(self):
        """Test that CnnPredictor correctly predicts on a file data."""
        predictor = CnnPredictor()

        for filename in SUPPORT_DIRNAME.glob("*.csv.gz"):
            pred, loss = predictor.predict(str(filename))
            print(f"Prediction: {pred} with loss: {conf} for a file")
            self.assertGreater(conf, 0.5)