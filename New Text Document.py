import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

dataset_train = pd.read_csv("F:\\Code practice\\titanic\\train.csv")
dataset_train = dataset_train.drop(labels = ["PassengerId","Name","Ticket","Cabin","Embarked"],axis = 1)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
genderLE = LabelEncoder()
dataset_train.iloc[:,2] = genderLE.fit_transform(dataset_train.iloc[:,2])
ohe=dataset_train.iloc[:,2].values
ohe=ohe.reshape(-1,1)
genderohe = OneHotEncoder(categorical_features = [0])
ohe = genderohe.fit_transform(ohe).toarray()
dataset_train = dataset_train.drop(labels = "Sex",axis = 1)

dataset_train = dataset_train.assign(Sex1 = ohe[:,0])
dataset_train = dataset_train.assign(Sex2 = ohe[:,1])

age = dataset_train.iloc[:,2].values
age = age.reshape(-1,1)
indices = []
for i in range(0,len(age)):
    if np.isnan(age[i]):
        indices.append(i)
    else:
        continue
fare = dataset_train.iloc[:,5].values
fare = fare.reshape(-1,1)

#For Regression items are to be removed vars == age_removed,fare_removed
age_removed = np.delete(age,indices,axis = None).reshape(-1,1)
fare_removed = np.delete(fare,indices,axis = None).reshape(-1,1)
removed_fare_elements = np.array(fare)[indices]

from sklearn.linear_model import LinearRegression
regressor_agefare = LinearRegression()
regressor_agefare.fit(fare_removed,age_removed)

age_pred = regressor_agefare.predict(removed_fare_elements)

for j in range(0,len(indices)):
    temp = indices[j]
    age[temp] = age_pred[j]
#Cleaning Test Set
dataset_test = pd.read_csv("F:\\Code practice\\titanic\\test.csv")
dataset_test = dataset_test.drop(labels = ["PassengerId","Name","Ticket","Cabin","Embarked"],axis = 1)
dataset_test.iloc[152, dataset_test.columns.get_loc('Fare')] = 26
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
genderLE_test = LabelEncoder()
dataset_test.iloc[:,1] = genderLE_test.fit_transform(dataset_test.iloc[:,1])
ohe_test=dataset_test.iloc[:,1].values
ohe_test=ohe_test.reshape(-1,1)
genderohe_test = OneHotEncoder(categorical_features = [0])
ohe_test = genderohe_test.fit_transform(ohe_test).toarray()
dataset_test = dataset_test.drop(labels = "Sex",axis = 1)

dataset_test = dataset_test.assign(Sex1 = ohe_test[:,0])
dataset_test = dataset_test.assign(Sex2 = ohe_test[:,1])


age_test = dataset_test.iloc[:,1].values
age_test = age_test.reshape(-1,1)
indices_test = []
for i in range(0,len(age_test)):
    if np.isnan(age_test[i]):
        indices_test.append(i)
    else:
        continue
fare_test = dataset_test.iloc[:,4].values
fare_test = fare_test.reshape(-1,1)

#For Regression items are to be removed vars == age_removed,fare_removed
age_removed_test = np.delete(age_test,indices_test,axis = None).reshape(-1,1)
fare_removed_test = np.delete(fare_test,indices_test,axis = None).reshape(-1,1)
removed_fare_elements_test = np.array(fare_test)[indices_test]
fare_removed_test[121,0] = 26
from sklearn.linear_model import LinearRegression
regressor_agefare_test = LinearRegression()
regressor_agefare_test.fit(fare_removed_test,age_removed_test)

age_pred_test = regressor_agefare_test.predict(removed_fare_elements_test)

for k in range(0,len(indices_test)):
    temp_test = indices_test[k]
    age_test[temp_test] = age_pred_test[k]

final_X_train = dataset_train.iloc[:,[1,2,3,4,5,6,7]].values
final_Y_train = dataset_train.iloc[:,0].values
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(final_X_train,final_Y_train)
final_X_test = dataset_test.iloc[:,:]
final_prediction = logistic.predict(final_X_test).reshape(-1,1)









