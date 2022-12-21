import json
import pickle

import numpy as np
import pandas as pd
import sns as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

from LogisticRegression import vectorize_df, features_list

with open("models/LogisticRegression.pkl", "rb") as f:
    model = pickle.load(f)

dataset_number=1
# Read the valuation json, preprocess it and run your model
with open(f'./dataset_{str(dataset_number)}_val.json') as file:
    raw_ds = json.load(file)
test_df = pd.json_normalize(raw_ds, max_level=2)

# Preprocess the validation dataset, remember that here you don't have the labels
test_df = vectorize_df(test_df)

# Predict with your model
X_new = test_df[features_list].to_numpy()
# Use the model to make predictions on new data
predictions = model.predict(X_new)


############ Save The predictions ############
test_type = 'label' # Options are ['label', 'attack_type']
enc = LabelEncoder()
np.savetxt(f'results/dataset_{str(dataset_number)}_{test_type}_result.txt', enc.fit_transform(predictions), fmt='%2d')
