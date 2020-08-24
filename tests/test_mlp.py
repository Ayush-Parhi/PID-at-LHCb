"""Tests for MLPmodel class."""
import os
from pathlib import Path
import unittest

from mlp_predictor import MlpPredictor

SUPPORT_DIRNAME = Path(__file__).parents[0].resolve() / "support"

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestMlpPredictor(unittest.TestCase):
    """Tests for the MlpPredictor class."""

    def test_filename(self):
        """Test that MlpPredictor correctly predicts on a file data."""
        predictor = MlpPredictor()

        for filename in SUPPORT_DIRNAME.glob("*.csv.gz"):
            pred, loss = predictor.predict(str(filename))
            print(f"Prediction: {pred} with loss: {conf} for a file data")
            self.assertGreater(conf, 0.5)