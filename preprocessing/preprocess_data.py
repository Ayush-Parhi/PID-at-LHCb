import pandas as pd
from preprocessing.utils import *

data = pd.read_csv("../data/training.csv.gz")
data['Class'] = get_class_ids(data.Label.values)
features = extract_features(data)