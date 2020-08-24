"""Define a 1-D CNN network function"""
import numpy as np 
import pandas as pd
from typing import Tuple
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from keras.models import Model, Sequential

def cnn(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
    layer_sizes: [128, 64]
) -> Model:
    """
    Create a simple one-dimensional convolutional neural network: fully-connected layerss, with softmax predictions.
    """
    num_classes = output_shape[0]

    model = Sequential()
    
    model.add(Conv1D(filters = 64, kernel_size = 2, activation ='tanh', input_shape = input_shape ))
    model.add(MaxPooling1D())
    model.add(Dropout(0.3))

    model.add(Flatten())
    for layer_size in layer_sizes:
        model.add(Dense(layer_size, activation='tanh'))
    model.add(Dense(num_classes, activation = 'softmax'))

    return model
