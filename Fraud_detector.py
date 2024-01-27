import pandas as pd
import numpy as np
data=pd.read_csv('C:\\Users\\Acer\\Downloads\\archive (1)\\PS_20174392719_1491204439457_log.csv')

data.head()

print(data.isnull().sum())

print(data.type.value_counts()) #uniquely evaluates the number of occurences of an attribute

type = data["type"].value_counts()
transactions = type.index
quantity = type.values

import plotly.express as px
figure = px.pie(data, 
                values=quantity, 
                names=transactions, 
                hole=0.5, 
                title="Distribution of Transaction Type")
figure.show()

#examining the correlation between different components
numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Select only numeric column
correlation=numeric_data.corr() #between data features and is fraud column
print(correlation["isFraud"].sort_values(ascending=False))

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5}) 
#we have converted all the string data type trandactions into numeric

data['isFraud']=data['isFraud'].map({0:'isFraud',1:'isnotFraud'}) #converting string value in fraud column to numeric 

data.head()

# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])

# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 9.0]])
print(model.predict(features))

features= np.array([[3,11668.14,11668.14,0.0]])
print(model.predict(features))

features = np.array([[2,9839.64,9839.64,0.0]])
print(model.predict(features))
