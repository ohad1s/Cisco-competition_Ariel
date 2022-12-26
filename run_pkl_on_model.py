import pickle
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder

# Set pandas to show all columns when you print a dataframe
pd.set_option('display.max_columns', None)
dataset_number = 1  # Options are [1, 2, 3, 4]
# test_type = 'label'  # Options are ['label', 'attack_type']
# with open(f'combined_datasets_for_students/dataset_{str(dataset_number)}_train.json') as file:
#     raw_ds = json.load(file)
# df = pd.json_normalize(raw_ds, max_level=2)



def categorize(row):
    if row['request.Attack_Tag'] == 'Benign':
        return 'Benign'
    return 'Malware'


# On these headers we will run a "simple" BOW
SIMPLE_HEADERS = ['request.headers.Accept-Encoding',
                  'request.method',
                  'request.headers.Accept-Language',
                  'request.headers.Sec-Fetch-Site',
                  'request.headers.Sec-Fetch-Mode',
                  'request.headers.Sec-Fetch-Dest',
                  'response.status',
                  'response.status_code'

                  ]

# On these headers we will run HashingVectorizer
COMPLEX_HEADERS = ['request.headers.User-Agent',
                   'request.headers.Set-Cookie',
                   'request.headers.Date',
                   'request.url',
                   'response.body',
                   'response.headers.Location',
                   'request.headers.Content-Length',
                   'request.headers.Cookie',
                   'response.headers.Set-Cookie'
                   ]

COLUMNS_TO_REMOVE = ['request.headers.Host',
                     'request.headers.Accept',
                     'request.headers.Connection',
                     'request.headers.Sec-Fetch-User',
                     'response.headers.Content-Type',
                     'request.body',
                     'response.status_code',
                     'request.headers.Set-Cookie',
                     'request.method',
                     'response.headers.Location',
                     'response.headers.Set-Cookie'
                     ]


# This is our main preprocessing function that will iterate over all of the chosen
# columns and run some feature extraction models
def vectorize_df(df):
    le = LabelEncoder()
    h_vec = HashingVectorizer(n_features=4)

    # Run LabelEncoder on the chosen features
    for column in SIMPLE_HEADERS:
        df[column] = le.fit_transform(df[column])

    # Run HashingVectorizer on the chosen features
    for column in COMPLEX_HEADERS:
        newHVec = h_vec.fit_transform(df[column])
        df[column] = newHVec.todense()

    # Remove some columns that may be needed.. (Or not, you decide)
    for column in COLUMNS_TO_REMOVE:
        df.drop(column, axis=1, inplace=True)
    return df




# In our example code we choose all the columns as our feature this can be the right or wrong way to approach the model, you choose.

with open("models/RandomForestImprovement.pkl", "rb") as f:
    model = pickle.load(f)

dataset_number = 1
# Read the valuation json, preprocess it and run your model
with open(f'combined_datasets_for_students/dataset_{str(dataset_number)}_val.json') as file:
    raw_ds = json.load(file)
test_df = pd.json_normalize(raw_ds, max_level=2)

# Preprocess the validation dataset, remember that here you don't have the labels

#
# test_df['request.Attack_Tag'] = test_df['request.Attack_Tag'].fillna('Benign')
# test_df['attack_type'] = test_df['request.Attack_Tag']
#
# test_df['label'] = test_df.apply(lambda row: categorize(row), axis=1)

# After finishing the arrangements we delete the irrelevant column
# test_df.drop('request.Attack_Tag', axis=1, inplace=True)
for column in test_df.columns[test_df.isna().any()].tolist():
    # df.drop(column, axis=1, inplace=True)
    test_df[column] = test_df[column].fillna('None')


features_list = test_df.columns.to_list()
# features_list.remove('label')
# features_list.remove('attack_type')


test_df = vectorize_df(test_df)
# Predict with your model
X_new = test_df[features_list].to_numpy()
# Use the model to make predictions on new data
predictions = model.predict(X_new)

############ Save The predictions ############
test_type = 'label'  # Options are ['label', 'attack_type']
enc = LabelEncoder()
np.savetxt(f'results/dataset_{str(dataset_number)}_{test_type}_result.txt', enc.fit_transform(predictions), fmt='%2d')
