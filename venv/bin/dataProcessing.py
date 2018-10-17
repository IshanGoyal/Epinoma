'''
Filename: dataProcessing.py
Objective: Add the excel data file into a Pandas dataframe and remove the excess columns
Input: Need to specify path to excel file
Output: Creates processed dataframe and stores in memory using Pickle package
'''



import pandas as pd
import pickle
from pandas import ExcelWriter
from pandas import ExcelFile

#IMPORTANT: Specify path here
df = pd.read_excel('KangZhang_et.all_2016_processed_data.xlsx', sheetname='hisq13_cf_healthy_liver_468mark')

print("Original Dimensions: " + str(df.shape))
print("Removing unwanted columns from data frame: ")

#Want to keep every third column (methylation values)
countVar = 0
for column in df:
    if(countVar-2 > 0 and (countVar-2)%3 != 0):
        df = df.drop(str(column),1)
    countVar = countVar+1


#Store new dataframe into memory using pickle
print("New Dimensions: " + str(df.shape))
print(df.head())
df.to_pickle('myDF.pickle')
del df
