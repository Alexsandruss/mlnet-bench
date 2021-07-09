from sys import argv
import pandas as pd
from sklearn.model_selection import train_test_split


original_data = pd.read_csv(argv[1])
data = pd.DataFrame()
data['f0'] = original_data['userId']
data['f1'] = original_data['movieId']
data['target'] = original_data['rating']

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
train_data.to_csv('data/movielens_train.csv', index=False)
test_data.to_csv('data/movielens_test.csv', index=False)
