import pandas as pd
import numpy as ny

#Trevor Henderson, Thai Thomas Vang
#Dataset from: https://www.kaggle.com/datasets/shraddha4ever20/titanic-datset

#Imports and finds the csv file
df=pd.read_csv('E:/Python Project  folder/titanic.csv')

#Making a copy of Original Sheet to edit
df1=df.copy()

#replace blank age with "Unknown" due to age either being known or unknown and not a mean or 0
df1[['age']]=df1[['age']].fillna("Unknown")

#drop rows that have all data missing (note this dataset titanic.csv has no missing data row)
df1.dropna(how='all')

#converting True/False to Yes/No for adult_male
df1['adult_male']=df1['adult_male'].map({True: 1, False: 0})

#Changing NaN in deck to Unknown for the same reason to not have nothing as a value for deck location
df1[['deck']]=df1[['deck']].fillna("Unknown")

#Changing alone column to yes to 1 and no to 0
df1['alone']=df1['alone'].map({True: 1, False: 0})

#Correcting alive status to capital yes to 1 and no to 0
df1['alive']=df1['alive'].map({"yes": 1, "no": 0})

#Correcting null cells in embarked and embark_town columns
df1[['embarked']]=df1[['embarked']].fillna("Unknown")
df1[['embark_town']]=df1[['embark_town']].fillna("Unknown")

#Rearranges columns to be next to their related ones
new_cols=['survived','pclass','alive','alone','sex','who','age','adult_male','class','deck','embarked','embark_town','sibsp','parch','fare']
df1=df1.reindex(columns=new_cols)

#Capitalizing and renaming columns
print(df1.rename(columns={'survived':'Survived','pclass':'PClass','alive':'Alive','alone':'Alone','sex':'Sex','who':'Who','age':'Age','adult_male':'Adult_Male','class':'Class','deck':'Deck','embarked':'Embarked','embark_town':'Embark_Town','sibsp':'Sibsp','parch':'Parch','fare':'Fare'},inplace=True))

#Getting rid of columns Sibsp and Parch due to bloat
df1=df1.drop('Sibsp',axis=1)
df1=df1.drop('Parch',axis=1)
#Prints out the entire csv
print(df1)

#shows cells that are empty/null
print(df1.isnull().sum())

#This code prints out and shows all different column names for easy reference
print(df1.columns)

df1.to_csv('Finished Titanic Dataset.csv', index=False)
