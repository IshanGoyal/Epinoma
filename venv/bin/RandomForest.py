import pandas as pd
import numpy as np
import pickle
from pandas import ExcelWriter
from pandas import ExcelFile

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_pickle('myDF.pickle')
print(" Dimensions: " + str(df.shape))
print(df.head())

targetLabels = list(df.columns.values)[:1] #PatientType
featureLabels = list(df.columns.values)[2:] #List includes all methylation markers
print("Target Labels: " + str(targetLabels)) #Target Labels
print("Feature Labels: " + str(featureLabels)) #Features Labels

targets = df.iloc[:, :1] #All of the patient labels
features = df.iloc[:,2:] #All of the biomarker methylation data

print("Patient Labels: ")
print(targets)
print("Patient Data: ")
print(features)
features=np.array(features)
print(features)

X_train, X_test, y_train, y_test = train_test_split(features,targets, test_size=0.25, random_state=0) #70% training 30% test

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

regressor = RandomForestRegressor()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

clf = RandomForestClassifier(n_estimators=20, n_jobs=2, random_state=0)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))



