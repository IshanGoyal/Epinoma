'''
Filename: RandomForestFeatureList.py
Objective: Runs the random forest algorithm on the Pandas dataframe
Input: Number of decision trees to use, number of features to output
Output: Classification statistics, confusion matrix, list of tuples (feature, feature weight)
'''


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from pandas import ExcelFile
from operator import itemgetter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

#Input: numEstimators - # of trees in the random forest that is built
#       numFeatures - # of features that are ouputted

def featurelist(numEstimators, numFeatures):
    df = pd.read_pickle('myDF.pickle')

    print(df.head())
    print(df.tail())

    targets = df.iloc[:, :1] #All of the patient labels (1 = diseased, 0 = healthy)
    features = df.iloc[:,2:] #All of the biomarker methylation data

    df['is_train'] = np.random.uniform(0,1,len(df)) <= 0.8

    train,test = df[df['is_train']==True], df[df['is_train']==False]

    featureLabels = list(df.columns.values)[2:-1] #List includes all methylation markers
    targetLabels = list(df.columns.values)[:1] #PatientType

    y = pd.factorize(train['PatientType'])[0]
    clf = RandomForestClassifier(n_estimators=numEstimators, n_jobs=2)
    clf.fit(train[featureLabels],y)
    clf.predict(test[featureLabels])
    clf.predict_proba(test[featureLabels])[0:10]
    featureImportance = list(zip(train[featureLabels], clf.feature_importances_))

    #Sort the list of tuples by the second value
    featureImportance.sort(key=itemgetter(0))
    featureImportance.sort(key=itemgetter(1),reverse=True)
    print("The following methylation markers are the top 30 contributors to the determination: ")
    print(featureImportance[:numFeatures])


    X_train, X_test, y_train, y_test = train_test_split(features,targets, test_size=0.2) #80% training 20% test

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    regressor = RandomForestRegressor(n_estimators=numEstimators, n_jobs=2)
    regressor.fit(X_train,y_train.values.ravel())
    y_pred = regressor.predict(X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    clf = RandomForestClassifier(n_estimators=numEstimators, n_jobs=2)
    clf.fit(X_train,y_train.values.ravel())
    y_pred=clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    print("Training Dataset Size: ")
    print(X_train.shape)
    print("Test Dataset Size: ")
    print(X_test.shape)

    #Summary statistics on classification
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test,y_pred))
    print("Classification Report: ")
    print(classification_report(y_test,y_pred))
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

    #Plot AUROC curve using matplotlib
    preds = probs[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("AUROCgraph.png")
    print("AUC:" + str(roc_auc))

    #Return the top features
    return(featureImportance[:30])


output = (list(featurelist(10,30)))
#Create a running list of all features generated and save in an output text file
with open('outputFeatures.txt','a+') as fp:
    fp.write('\n'.join('{} {}'.format(item[0],item[1])for item in output))
    fp.write('\n')




