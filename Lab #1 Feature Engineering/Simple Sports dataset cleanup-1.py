import pandas as pd
import numpy as np

#Thai Thomas Vang, Trevor Henderson
#Dataset from: https://www.kaggle.com/datasets/ttv9129/simple-sports-dataset/data

#Imports and finds the csv file
df=pd.read_csv('E:/Python Project  folder/simple_sports_dirty_300.csv')

df1=df.copy()

#Filling pts, assists, and rebounds with mean value
df1[['pts','assists','rebounds']]=df1[['pts','assists','rebounds']].fillna(value=df1[['pts','assists','rebounds']].mean())

#Filling no team name with Unknown
df1[['team']]=df1[['team']].fillna("Unknown")

#Prints the csv
print(df1)

#Prints and shows information about the cells
print(df1.info)

#shows null/empty cells
print(df1.isnull().sum())

df1.to_csv('Finished Sports Dataset.csv', index=False)
