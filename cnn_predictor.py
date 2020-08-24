from typing import Tuple, Union

import numpy as np
import pandas as pd

from models.cnn_model import CnnModel
from preprocessing.preprocess_data import features
from preprocessing.utils import label_class_correspondence, scale_data



class CnnPredictor:
    """Given csv or xlsx file of data from LHCb, predicts the particle type."""

    def __init__(self):
        self.model = CnnModel()
        self.model.load_weights()

    def predict(self, filename):
        """Predict on the data file."""
        if 'csv' in filename:
            data = pd.read_csv(filename)
        elif 'xlsx' in filename:
            data = pd.read_xlsx(filename)
        
        data = scale(data, features).values
        pred, _ = self.model.predict_mlp(data.reshape(-1, 49, 1))

        prediction = pandas.DataFrame({'ID': ids})
        for name in ['Ghost', 'Electron', 'Muon', 'Pion', 'Kaon', 'Proton']:
            prediction[name] = pred[:, label_class_correspondence[name]]
        prediction.to_csv('predictions.csv.gz', index=False, float_format='%.5f', compression="gzip")
        return FileLink('predictions.csv.gz')
