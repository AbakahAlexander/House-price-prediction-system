import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

house_data = pd.read_csv('Bangalore.csv')
house_data.head()
print(house_data.head())
house_data.tail()
print(house_data.tail())

print(house_data.shape)
house_data.info()
print(house_data.info())
house_data.isnull()
print(house_data.isnull())
house_data.describe()
print(house_data.describe())

x = house_data.drop(columns='Price', axis=1)

y = house_data['Price']

print(x)

print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, y, test_size=0.2, stratify=y, random_state=1)

print(x.shape, x_train.shape, x_test.shape),

model = LogisticRegression()

model.fit(x_train, y_train)

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy on Training data ; ', training_data_accuracy)