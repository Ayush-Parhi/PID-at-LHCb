import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

label_class_correspondence = {'Electron': 0, 'Ghost': 1, 'Kaon': 2, 'Muon': 3, 'Pion': 4, 'Proton': 5}
class_label_correspondence = {0: 'Electron', 1: 'Ghost', 2: 'Kaon', 3: 'Muon', 4: 'Pion', 5: 'Proton'}

def get_class_ids(labels):
    """
    Convert particle type names into class ids.
    Parameters:
    -----------
    labels : array_like
        Array of particle type names ['Electron', 'Muon', ...].
    Return:
    -------
    class ids : array_like
        Array of class ids [1, 0, 3, ...].
    """
    return np.array([label_class_correspondence[alabel] for alabel in labels])

def extract_features(data):
    """
    Extract feature columns from the data.
    Parameters:
    -----------
    data : pandas dataframe

    Return:
    -------
    features : list_like
        list of feature columns.
    """
    features = list(set(data.columns) - {'Label', 'Class'})
    return features

def scale_data(data, features):
    """
    Scales the data according to standard deviation and mean.
    Parameters:
    -----------
    data : pandas dataframe
    features: list
        list of features extracted from the data.

    Return:
    -------
    scaled_data : numpy array
        array of scaled data.
    """
    scalar = StandardScaler()
    scaled_data = scalar.fit_transform(data[features].values)
    return scaled_data
