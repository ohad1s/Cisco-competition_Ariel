# data_set_4_attack_type_sol

# **############# Imports, settings and first dataset view ###############**


# Imports, settings and first dataset view
import pandas as pd
import seaborn as sns
import numpy as np
import json

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

# Set pandas to show all columns when you print a dataframe
pd.set_option('display.max_columns', None)

# Global setting here you choose the dataset number and classification type for the model
dataset_number = 4  # Options are [1, 2, 3, 4]
test_type = 'attack_type'  # Options are ['label', 'attack_type']

# Read the json and read it to a pandas dataframe object, you can change these settings
with open(f'combined_datasets_for_students/dataset_{str(dataset_number)}_train.json') as file:
    raw_ds = json.load(file)
data = pd.json_normalize(raw_ds, max_level=2)

df = data.copy()
df['request.Attack_Tag'] = df['request.Attack_Tag'].fillna('Benign')
df['request.headers.Set-Cookie'] = df['request.headers.Set-Cookie'].fillna("None")
df['request.headers.Sec-Fetch-Site'] = df['request.headers.Sec-Fetch-Site'].fillna("None")
df['request.headers.Sec-Fetch-Dest'] = df['request.headers.Sec-Fetch-Dest'].fillna("None")
df['attack_type'] = df['request.Attack_Tag']
df['request.Attack_Tag'].value_counts()

# df['request.Attack_Tag'] = df['request.Attack_Tag'].fillna('Benign')
# df['attack_type'] = df['request.Attack_Tag']


# This function will be used in the lambda below to iterate over the label columns 
# You can use this snippet to run your own lambda on any data with the apply() method
def categorize(row):
    if row['request.Attack_Tag'] == 'Benign':
        return 0
    if row['request.Attack_Tag'] == 'Cookie Injection':
        return 1
    if row['request.Attack_Tag'] == 'Directory Traversal':
        return 2
    if row['request.Attack_Tag'] == 'LOG4J':
        return 3
    if row['request.Attack_Tag'] == 'Log Forging':
        return 4
    if row['request.Attack_Tag'] == 'RCE':
        return 5
    if row['request.Attack_Tag'] == 'SQL Injection':
        return 6
    if row['request.Attack_Tag'] == 'XSS':
        return 7
    
# {'Benign': 0, 'Cookie Injection': 1, 'Directory Traversal': 2, 'LOG4J': 3, 'Log Forging': 4, 'RCE': 5, 'SQL Injection': 6, 'XSS': 7}
    # 
    

df['label'] = df.apply(lambda row: categorize(row), axis=1)

# After finishing the arrangements we delete the irrelevant column
df.drop('request.Attack_Tag', axis=1, inplace=True)

df

df['label'].value_counts()

# Remove all NAN columns or replace with desired string
# This loop iterates over all of the column names which are all NaN
for column in df.columns[df.isna().any()].tolist():
    # df.drop(column, axis=1, inplace=True)
    df[column] = df[column].fillna('None')

# If you want to detect columns that may have only some NaN values use this:
# df.loc[:, df.isna().any()].tolist()


# Setting features for further feature extraction by choosing columns
# Some will be "simply" encoded via label encoding and others with HashingVectorizer

# On these headers we will run a "simple" BOW
SIMPLE_HEADERS = [
    #   'request.headers.Host',
    #   'request.headers.Accept',
    #   'request.headers.Connection',
    #   'request.headers.Sec-Fetch-User',
    # 'response.headers.Content-Type',
    #   'request.body',
    #   'response.headers.Content-Length',
    # 'request.headers.Accept-Encoding',
    #   'request.method',
    # 'request.headers.Accept-Language',
    # 'request.headers.Sec-Fetch-Site',
    #   'request.headers.Sec-Fetch-Mode',
    # 'request.headers.Sec-Fetch-Dest',
    #   'response.status',
    #   'response.status_code'
]

# On these headers we will run HashingVectorizer
COMPLEX_HEADERS = [
                    # 'request.headers.User-Agent',
                   #    'request.headers.Set-Cookie',
                   #    'request.headers.Date',
                   #    'request.url',
                #    'response.body',
                #    'response.headers.Location',
                #    'request.headers.Content-Length',
                #    'request.headers.Cookie',
                #    'response.headers.Set-Cookie'
                   ]

COLUMNS_TO_REMOVE = ['request.headers.Host', 'request.headers.User-Agent',
       'request.headers.Accept-Encoding', 'request.headers.Accept',
       'request.headers.Connection', 'request.headers.Sec-Ch-Ua-Platform',
       'request.headers.Sec-Ch-Ua-Mobile', 'request.headers.Accept-Language',
       'request.headers.Sec-Fetch-Site', 'request.headers.Sec-Fetch-Mode',
       'request.headers.Cache-Control', 'request.headers.Sec-Fetch-User',
       'request.headers.Sec-Fetch-Dest', 'request.headers.Set-Cookie',
       'request.headers.Date', 'request.method', 'request.url', 'request.body',
       'response.status', 'response.headers.Content-Type',
       'response.headers.Content-Length', 'response.status_code',
       'response.body', 'response.headers.Location',
       'request.headers.Cookie', 'request.headers.Content-Length',
       'response.headers.Set-Cookie',
       'request.headers.Upgrade-Insecure-Requests']

# This is our main preprocessing function that will iterate over all of the chosen 
# columns and run some feature extraction models
def vectorize_df(df):
    cookie_injection = []
    LOG4J =[]
    log_forging = []
    SQL_Injection = []
    RCE_Injection = []
    Directory_Traversal =[]
    XSS =[]
    for index, row in df.iterrows():
        if "%20or%20" in row['request.url'] or 'SELECT' in row['request.url']:
            SQL_Injection.append(1)
        else:
            SQL_Injection.append(0)
        if "/cookielogin" in row['request.url']:
            cookie_injection.append(1)
        else:
            cookie_injection.append(0)
        if "%20user%20" in row['request.url']:
            log_forging.append(1)
        else:
            log_forging.append(0) 
        if '/passwd.txt' in row['request.url'] or "/windows.ini.txt" in row['request.url'] or'/passwords.txt' in row['request.url'] or '/secrets.txt' in row['request.url'] :
            Directory_Traversal.append(1)
        else:
            Directory_Traversal.append(0)
        if "script" in row['request.url'] or "get_flashed_messages" in row['request.url']:
            RCE_Injection.append(1)
        else:
            RCE_Injection.append(0)
        if '{jndi' in row['request.headers.Sec-Fetch-Site']or '{jndi' in row['request.headers.Accept-Encoding'] or'{jndi' in row["request.headers.Set-Cookie"] or (row['request.headers.Sec-Fetch-Dest'] != "document" and row['request.headers.Sec-Fetch-Dest'] != "None") :
            LOG4J.append(1)
        else:
            LOG4J.append(0)
        if "forum" in row['request.url']:
            XSS.append(1)
        else:
            XSS.append(0)
        
    df["SQL_Injection"] = SQL_Injection
    df["LOG4J"] = LOG4J
    df["cookie_injection"] = cookie_injection
    df["log_forging"] = log_forging
    df["RCE_Injection"] = RCE_Injection
    df["Directory_Traversal"] = Directory_Traversal
    df["XSS"] = XSS
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


df = vectorize_df(df)

# Memory check (For large datasets sometimes the dataframe will exceed the computers resources)
df.info(memory_usage="deep")

# Choose the right features
# In our example code we choose all the columns as our feature this can be the right or wrong way to approach the model, you choose.

features_list = df.columns.to_list()
features_list.remove('label')
features_list.remove('attack_type')
print(features_list)

# Recheck all datatype before training to see we don't have any objects in our features
# In this example our model must get features containing only numbers so we recheck to see if we missed anything during preprocessing
df.dtypes

# Data train and test split preparations. Here we will insert our feature list and label list.
# Afterwards the data will be trained and fitted on the amazing XGBoost model
# X_Train and y_Train will be used for training
# X_test and y_test.T will be used for over fitting checking and overall score testing

# We convert the feature list to a numpy array, this is required for the model fitting
X = df[features_list].to_numpy()

# This column is the desired prediction we will train our model on
y = np.stack(df[test_type])

# We split the dataset to train and test according to the required ration
# Do not change the test_size -> you can change anything else
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1765, random_state=42, stratify=y)

# We print the resulted datasets and count the difference 
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
counter = Counter(y)
counter

from sklearn import tree
# clf = tree.DecisionTreeClassifier()
from sklearn.linear_model import LogisticRegression

# We choose our model of choice and set it's hyper parameters you can change anything
clf = RandomForestClassifier()
# clf = LogisticRegression()

# Train Model
clf.fit(X_train, y_train)

# Check data balance and variety
print(sorted(Counter(y_train).items()))

# We print our results
sns.set(rc={'figure.figsize': (15, 8)})
predictions = clf.predict(X_test)
true_labels = y_test
list_not_eq =[]
for i in range(len(predictions)):
    if predictions[i] != true_labels[i]:
        print(f'predictions {predictions[i]}, true_labels {true_labels[i]}')
        list_not_eq.append(i)
print(list_not_eq)
print(len(list_not_eq))
cf_matrix = confusion_matrix(true_labels, predictions)
clf_report = classification_report(true_labels, predictions, digits=5)
heatmap = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g',
                      xticklabels=np.unique(true_labels),
                      yticklabels=np.unique(true_labels))

# The heatmap is cool but this is the most important result
print(clf_report)

# Now it's your turn, use the model you have just created :)

# Read the valuation json, preprocess it and run your model 
with open(f'combined_datasets_for_students/dataset_{str(dataset_number)}_val.json') as file:
    raw_ds = json.load(file)
test_df = pd.json_normalize(raw_ds, max_level=2)
test_df


for column in test_df.columns[test_df.isna().any()].tolist():
    # df.drop(column, axis=1, inplace=True)
    test_df[column] = test_df[column].fillna('None')

# Preprocess the validation dataset, remember that here you don't have the labels
test_df = vectorize_df(test_df)
# Predict with your model
X = test_df[features_list].to_numpy()
predictions = clf.predict(X)
predictions

# Save your preditions
enc = LabelEncoder()
np.savetxt(f'results/dataset_{str(dataset_number)}_{test_type}_result.txt', enc.fit_transform(predictions), fmt='%2d')

