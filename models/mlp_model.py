"""MLPModel class."""
from typing import Callable, Dict, Tuple

import numpy as np

from sklearn.metrics import log_loss
from models.base import Model
#from models.base.Model.fit import validation_data_scaled
from preprocessing.preprocess_data import features, data
from networks.mlp import mlp


class MlpModel(Model):

    def __init__(
        self,
        dataset: type = data,
        network_fn: Callable = mlp,
        network_args: Dict = None,
        training_args: Dict = {'verbose':True, 'batch_size':1024, 'epochs':150}

    ):
        super().__init__(dataset, network_fn, network_args, training_args)

    def predict_mlp(self, data):
        pred = self.network.predict(data)
        loss = log_loss(data.Class.Values, pred)
        return pred, loss



