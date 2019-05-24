# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 20:32:38 2019

@author: Prakash
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
dataset1 = pd.read_csv("test.csv")

X = dataset.iloc[:, 9].values
y = dataset.iloc[:, 5].values
X_test = []
indices=[]
for i in range(0,len(y)):
    if (np.isnan(y[i])):
        X_test.append(X[i])
        indices.append(i)
        
    else:
        continue
    
X = np.delete(X,indices,axis = None)    
y = np.delete(y,indices,axis = None)    

X_test = np.array(X_test)
X_test = X_test.reshape(-1,1)
X = np.array(X)
X = X.reshape(-1,1)

Xt = dataset.iloc[:, 8].values
yt = dataset.iloc[:, 4].values
X_testt = []
indicest=[]
for j in range(0,len(yt)):
    if (pd.isnull(yt[j])):
        X_testt.append(Xt[j])
        indicest.append(j)
        
    else:
        continue
    
Xt = np.delete(Xt,indicest,axis = None)    
yt = np.delete(yt,indicest,axis = None)    

X_testt = np.array(X_testt)
X_testt = X_testt.reshape(-1,1)
Xt = np.array(Xt)
Xt = Xt.reshape(-1,1)
# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X= sc_X.fit_transform(X)



# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
j=0
for i in range(0,891):
    if (np.isnan(dataset.iloc[i,5])):
        dataset.iloc[i,5]=y_pred[j]
        j=j+1
       
    else:
        continue
    
dataset.drop(["PassengerId","Name","Ticket","Cabin","Embarked"],axis = 1,inplace = True)

O= dataset.iloc[:,0].values
I= dataset.iloc[:,[1,2,3,4,5,6]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_I = LabelEncoder()
I[:, 1] = labelencoder_I.fit_transform(I[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
I = onehotencoder.fit_transform(I).toarray()

from sklearn.svm import SVC
classifier = SVC(kernel = "rbf")
classifier.fit(I,O)

test = pd.read_csv("test.csv")
O_pred = classifier.predict(test)





