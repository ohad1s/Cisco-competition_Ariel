############# Imports, settings and first dataset view ###############
import csv
import pandas as pd
import seaborn as sns
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Set pandas to show all columns wgi hen you print a dataframe
from tabulate import tabulate

pd.set_option('display.max_columns', None)

# Global setting here you choose the dataset number and classification type for the model
dataset_number = 1  # Options are [1, 2, 3, 4]
test_type = 'label'  # Options are ['label', 'attack_type']

# Read the json and read it to a pandas dataframe object, you can change these settings
with open(f'combined_datasets_for_students/dataset_{str(dataset_number)}_train.json') as file:
    raw_ds = json.load(file)
df = pd.json_normalize(raw_ds, max_level=2)

############## Basic dataset label arrangements ###############

# Fill the black attack tag lines with "Benign" string
df['request.Attack_Tag'] = df['request.Attack_Tag'].fillna('Benign')
df['attack_type'] = df['request.Attack_Tag']


# This function will be used in the lambda below to iterate over the label columns
# You can use this snippet to run your own lambda on any data with the apply() method
def categorize(row):
    if row['request.Attack_Tag'] == 'Benign':
        return 'Benign'
    return 'Malware'


df['label'] = df.apply(lambda row: categorize(row), axis=1)

# After finishing the arrangements we delete the irrelevant column
df.drop('request.Attack_Tag', axis=1, inplace=True)

# Remove all NAN columns or replace with desired string
# This loop iterates over all of the column names which are all NaN
for column in df.columns[df.isna().any()].tolist():
    # df.drop(column, axis=1, inplace=True)
    df[column] = df[column].fillna('None')

# Setting features for further feature extraction by choosing columns
# Some will be "simply" encoded via label encoding and others with HashingVectorizer

# On these headers we will run a "simple" BOW
SIMPLE_HEADERS = ['request.headers.Accept-Encoding',
                  'request.headers.Connection',
                  'request.headers.Host',
                  'request.headers.Accept',
                  'request.method',
                  'request.headers.Accept-Language',
                  'request.headers.Sec-Fetch-Site',
                  'request.headers.Sec-Fetch-Mode',
                  'request.headers.Sec-Fetch-Dest',
                  'request.headers.Sec-Fetch-User',
                  'response.status',
                  ]

# On these headers we will run HashingVectorizer
COMPLEX_HEADERS = ['request.headers.User-Agent',
                   'request.headers.Set-Cookie',
                   'request.headers.Date',
                   'request.url',
                   'response.headers.Content-Type',
                   'response.body',
                   'response.headers.Location',
                   'request.headers.Content-Length',
                   'request.headers.Cookie',
                   'response.headers.Set-Cookie'
                   ]

# COLUMNS_TO_REMOVE = ['request.body',
#                      'response.headers.Content-Length',
#                      'request.headers.Date']

COLUMNS_TO_REMOVE = ['request.headers.Host', 'request.headers.Date', 'request.method',
                     'request.headers.Accept-Language']


# This is our main preprocessing function that will iterate over all of the chosen
# columns and run some feature extraction models
def vectorize_df(df):
    le = LabelEncoder()
    h_vec = HashingVectorizer(n_features=6)

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


df = vectorize_df(df)
##################################
features_list = df.columns.to_list()
features_list.remove('label')
features_list.remove('attack_type')
##################### create encode from string values to float ########################
from sklearn.preprocessing import OneHotEncoder

# columns_to_encode = ['request.headers.Host', 'request.headers.User-Agent', 'request.headers.Accept-Encoding',
#                      'request.headers.Accept', 'request.headers.Connection', 'request.headers.Accept-Language',
#                      'request.headers.Sec-Fetch-Site', 'request.headers.Sec-Fetch-Mode', 'request.headers.Sec-Fetch-User',
#                      'request.headers.Sec-Fetch-Dest', 'request.headers.Set-Cookie', 'request.headers.Date',
#                      'request.method', 'request.url', 'request.headers.Cookie','request.body']


columns_to_encode = ['request.headers.User-Agent', 'request.headers.Accept-Encoding',
                     'request.headers.Accept', 'request.headers.Connection',
                     'request.headers.Sec-Fetch-Site', 'request.headers.Sec-Fetch-Mode',
                     'request.headers.Sec-Fetch-User',
                     'request.headers.Sec-Fetch-Dest', 'request.headers.Set-Cookie',
                     'request.url', 'request.headers.Cookie', 'request.body']

X = pd.get_dummies(df[features_list], columns=columns_to_encode)

# This column is the desired prediction we will train our model on
y = np.stack(df[test_type])
y = pd.get_dummies(y, columns=columns_to_encode)
# print(tabulate(X.head(), headers = 'keys', tablefmt = 'psql'))
# breakpoint()

############# Decision Tree : ################
from sklearn.tree import DecisionTreeClassifier

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the model on the training data
clf.fit(X_train, y_train)

######### analyze the results: #########
true_labels = y_test
predictions = clf.predict(X_test)
# cf_matrix = confusion_matrix(true_labels, predictions)
clf_report = classification_report(true_labels, predictions, digits=5)
# heatmap = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g',
#                       xticklabels=np.unique(true_labels),
#                       yticklabels=np.unique(true_labels))

# The heatmap is cool but this is the most important result
print(clf_report)

accuracy = accuracy_score(true_labels, predictions)
print(f'Model accuracy: {accuracy:.2f}')
############# save the model to pickle file ##############
import pickle

# Save the model to a file
with open("models/DecisionTree.pkl", "wb") as f:
    pickle.dump(clf, f)

############## Save the prediction of the 0.2 test data to a csv file and #################
# # Create a data frame with the predictions and true labels
# df = pd.DataFrame({'predictions': predictions, 'true_labels': true_labels})
#
# # Write the data frame to a CSV file
# df.to_csv('csvs/DecisionTree.csv', index=False)

# ############### Save the report of the 0.2 test data to a csv file and #################
# # Split the classification report into lines
# lines = clf_report.split('\n')
#
# # Open a file for writing
# with open('csvs/reports/DecisionTreeReport.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     # Write the header row
#     writer.writerow(lines[0].split())
#     # Write the data rows
#     for line in lines[2:-1]:
#         writer.writerow(line.split())
