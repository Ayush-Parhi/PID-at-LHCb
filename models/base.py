"""Model class, to be extended by specific types of models."""
from pathlib import Path
from typing import Callable, Dict, Optional

from sklearn.model_selection import train_test_split
from preprocessing.preprocess_data import features
from preprocessing.utils import scale_data
from keras.models import Model as KerasModel
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np


DIRNAME = Path(__file__).parents[1].resolve() / "weights"


class Model:
    """Base class, to be subclassed by predictors for specific type of data."""

    def __init__(
        self,
        dataset,
        network_fn: Callable[..., KerasModel],
        network_args: Dict = None,
        training_args: Dict = None,
        
    ):
        self.name = f"{self.__class__.__name__}_{network_fn.__name__}"

        if training_args is None:
            training_args = {}
        self.training = training_args

        self.dataset = dataset

        if network_args is None:
            network_args = {}
        self.network = network_fn(self.data.input_shape, self.data.output_shape, **network_args)
        self.network.summary()


        self.batch_augment_fn: Optional[Callable] = None
        self.batch_format_fn: Optional[Callable] = None

    @property
    def image_shape(self):
        return self.data.input_shape

    @property
    def weights_filename(self) -> str:
        DIRNAME.mkdir(parents=True, exist_ok=True)
        return str(DIRNAME / f"{self.name}_weights.h5")

    def fit(
        self, dataset, **training_args 
    ):
        if callbacks is None:
            callbacks = []

        self.network.compile(loss=self.loss(), optimizer=self.optimizer(), metrics=self.metrics())

        training_data, validation_data = train_test_split(data, random_state=11, train_size=0.90)
        training_data_scaled = scale_data(training_data, features)
        validation_data_scaled = scale_data(validation_data, features)

        if network_fn == mlp:
            training_data_scaled, validation_data_scaled = training_data_scaled, validation_data_scaled

        elif network_fn == cnn:
            training_data_scaled, validation_data_scaled = training_data_scaled.reshape(-1, 49, 1), validation_data_scaled.reshape(-1, 49, 1)        

        self.network.fit(
            training_data_scaled,
            to_categorical(training_data.Class.values),
            validation_split = 0.01,
            **training_args  
            
        )

    def evaluate(self, x: np.ndarray, y: np.ndarray, batch_size: int = 16, _verbose: bool = False):
        # pylint: disable=unused-argument
        data = validation_data  # Use a small batch size to use less memory
        preds = self.network.predict(data)
        return preds

    def loss(self):  # pylint: disable=no-self-use
        return "categorical_crossentropy"

    def optimizer(self):  # pylint: disable=no-self-use
        return Adam()

    def metrics(self):  # pylint: disable=no-self-use
        return ["accuracy"]

    def load_weights(self):
        self.network.load_weights(self.weights_filename)

    def save_weights(self):
        self.network.save_weights(self.weights_filename)