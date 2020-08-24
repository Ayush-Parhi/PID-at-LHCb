"""Define a mlp network function"""
import numpy as np 
import pandas as pd
from typing import Tuple
from keras.layers import Dense, Flatten
from keras.models import Model, Sequential

def mlp(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    layer_sizes: [100, 50],
    num_layers: int = 2
) -> Model:
    """
    Create a simple multi-layer perceptron: fully-connected layerss, with softmax predictions.
    Creates num_layers layers.
    """
    num_classes = output_shape[0]

    model = Sequential()
    model.add(Flatten(input_shape = input_shape))
    for layer_size in layer_sizes:
        model.add(Dense(layer_size, activation='tanh'))
    model.add(Dense(num_classes, activation = 'softmax'))

    return model

