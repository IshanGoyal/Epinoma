'''
Filename: LassoMethod.py
Objective: Run the LASSO regularization method on the dataset to produce list of significant features
Input: Need to specify the alpha level of each LASSO run
Output: Summary statistics for each given alpha value, visualization of alpha level and coefficients of selected features, list of features and their coefficients
'''


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
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
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression

df = pd.read_pickle('myDF.pickle')

targetLabels = list(df.columns.values)[:1] #PatientType
featureLabels = list(df.columns.values)[2:] #List includes all methylation markers

targets = df.iloc[:, :1] #All of the patient labels (1 = diseased, 0 = healthy)
features = df.iloc[:,2:] #All of the biomarker methylation data


featureLabels = list(df.columns.values)[2:-1] #List includes all methylation markers
#print("Feature Labels: " + str(featureLabels)) #Features Labels
targetLabels = list(df.columns.values)[:1] #PatientType
#print("Target Labels: " + str(targetLabels)) #Target Labels


X_train, X_test, y_train, y_test = train_test_split(features,targets, test_size=0.2) #80% training 20% test
X_train = X_train.values #convert all data frames into numpy arrays for lasso
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values
y_train = y_train.flatten() #need to make np array into a vector
y_test = y_test.flatten() #need to make np array into a vector

# Create an empty data frame and add Feature Name column
featurelist = pd.DataFrame()
featurelist['Feature Name'] = featureLabels
alphas = [0.0001, 0.01, 0.05, 0.1]
# For each alpha value in the list of alpha values,
for alpha in alphas:
    # Create a lasso regression with that alpha value,
    lasso = Lasso(alpha=alpha)

    # Fit the lasso regression
    lasso.fit(X_train, y_train)

    # Create a column name for that alpha value
    column_name = 'Alpha = %f' % alpha

    # Create a column of coefficient values
    lasso.coef_ = (np.delete(lasso.coef_,-1)) #Remove last entry due to accomodate for change in size
    featurelist[column_name] = lasso.coef_

#Sort the features data
featurelist['Alpha = 0.010000'] = featurelist['Alpha = 0.010000'].astype('float')
featurelist = featurelist.sort_values(by=['Alpha = 0.010000'], ascending=[False])

lasso = Lasso() #default alpha =1
lasso.fit(X_train,y_train)
train_score=lasso.score(X_train,y_train)
test_score=lasso.score(X_test,y_test)
coeff_used = np.sum(lasso.coef_!=0)
print("training score for alpha=1:", train_score)
print("test score for alpha=1: ", test_score)
print("number of features used for alpha=1: ", coeff_used)
lasso01 = Lasso(alpha=0.1, max_iter=10e5) #alpha =0.1
lasso01.fit(X_train, y_train)
train_score = lasso01.score(X_train, y_train)
test_score = lasso01.score(X_test, y_test)
coeff_used = np.sum(lasso01.coef_ != 0)
print("training score for alpha=0.1:", train_score)
print("test score for alpha=0.1: ", test_score)
print("number of features used for alpha=0.1: ", coeff_used)
lasso005 = Lasso(alpha=0.05, max_iter=10e5) #alpha = 0.05
lasso005.fit(X_train,y_train)
train_score=lasso005.score(X_train,y_train)
test_score=lasso005.score(X_test,y_test)
coeff_used = np.sum(lasso005.coef_!=0)
print("training score for alpha=0.05:", train_score)
print("test score for alpha=0.05: ", test_score)
print("number of features used for alpha=0.05: ", coeff_used)
lasso001 = Lasso(alpha=0.01, max_iter=10e5) #alpha = 0.01
lasso001.fit(X_train,y_train)
train_score001=lasso001.score(X_train,y_train)
test_score001=lasso001.score(X_test,y_test)
coeff_used001 = np.sum(lasso001.coef_!=0)
print("training score for alpha=0.01:", train_score001)
print("test score for alpha =0.01: ", test_score001)
print("number of features used for alpha =0.01:", coeff_used001)
lasso00001 = Lasso(alpha=0.0001, max_iter=10e5) #alpha = 0.001
lasso00001.fit(X_train,y_train)
train_score00001=lasso00001.score(X_train,y_train)
test_score00001=lasso00001.score(X_test,y_test)
coeff_used00001 = np.sum(lasso00001.coef_!=0)
print("training score for alpha=0.0001:", train_score00001)
print("test score for alpha =0.0001: ", test_score00001)
print("number of features used for alpha =0.0001:", coeff_used00001)
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_train_score=lr.score(X_train,y_train)
lr_test_score=lr.score(X_test,y_test)
print("LR training score:", lr_train_score)
print("LR test score: ", lr_test_score)
plt.subplot(1,2,1)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.subplot(1,2,2)
plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\alpha = 1$',zorder=7) # alpha here is for transparency
plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\alpha = 0.01$') # alpha here is for transparency
plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\alpha = 0.0001$') # alpha here is for transparency
plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc=4)
plt.tight_layout()
plt.savefig("LASSOAnalysis.png")

# Return the dataframe with features and lasso coefficients
print(featurelist.head(30))