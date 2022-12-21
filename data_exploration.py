import json
import pandas as pd
import seaborn as sns
from IPython.display import display
from tabulate import tabulate
import matplotlib.pyplot as plt

dataset_number = 1

with open(f'combined_datasets_for_students/dataset_{str(dataset_number)}_train.json') as file:
    raw_ds = json.load(file)
df = pd.json_normalize(raw_ds, max_level=2)

# print(tabulate(df.head(), headers = 'keys', tablefmt = 'psql'))

# sns.histplot(df['response.headers.Content-Length'])
# sns.histplot(df['response.status_code'])
# plt.show()

# string_column = df['request.headers.Accept-Encoding'] # good feature to check!!
string_column = df['request.headers.Sec-Fetch-User']

# Count the number of times each unique value appears in the column
value_counts = string_column.value_counts(normalize=True)

# print(value_counts)
# value_counts.plot(kind='bar')
# plt.show()
