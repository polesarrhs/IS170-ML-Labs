import pandas as pd
import numpy as np

#Thai Thomas Vang, Trevor Henderson
#Dataset from: https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training?resource=download

#Imports and finds the csv file
df=pd.read_csv('C:/Users/Trevor/Documents/IS 170/dirty_cafe_sales.csv')

#Making a copy of Original Sheet to edit
df1=df.copy()


#drop transaction id column
df1=df1.drop(columns=['Transaction ID'])

#filling empty cells with a value and then replacing strings with 0.0 float value
df1['Total Spent'] = pd.to_numeric(df1['Total Spent'], errors='coerce')

#filling empty cells with a value and then replacing strings with 0.0 float value
df1['Quantity'] = pd.to_numeric(df1['Quantity'], errors='coerce')

#Convert any non-numeric values to NaN
df1['Price Per Unit'] = pd.to_numeric(df1['Price Per Unit'], errors='coerce')

# Using median to avoid outliers
item_price_map = df1.groupby('Item')['Price Per Unit'].median().to_dict()
print("Price mapping:")
print(item_price_map)

#Fill missing values
mask = df1['Price Per Unit'].isna()
df1.loc[mask, 'Price Per Unit'] = df1.loc[mask, 'Item'].map(item_price_map)

#check if Cake rows now have prices
print("\nCake rows:")
print(df1[df1['Item'] == 'Cake'][['Item', 'Quantity', 'Price Per Unit', 'Total Spent']])


# Fill Price Per Unit based on Item
item_price_map = df1.groupby('Item')['Price Per Unit'].median()
df1['Price Per Unit'] = df1['Price Per Unit'].fillna(df1['Item'].map(item_price_map))

# Calculate missing values
df1['Total Spent'] = df1['Total Spent'].fillna(df1['Quantity'] * df1['Price Per Unit'])
df1['Quantity'] = df1['Quantity'].fillna(df1['Total Spent'] / df1['Price Per Unit'])
df1['Price Per Unit'] = df1['Price Per Unit'].fillna(df1['Total Spent'] / df1['Quantity'])

# Fill any remaining NaN with 0
df1[['Total Spent', 'Quantity', 'Price Per Unit']] = df1[['Total Spent', 'Quantity', 'Price Per Unit']].fillna(0)

# Verify
print(df1[['Total Spent', 'Quantity', 'Price Per Unit']].isnull().sum())

#Filling Payment Method and Location blanks or NaN to Unknown
df1=df1.replace(["ERROR","UNKNOWN","Error","Unknown"],pd.NA)
df1['Transaction Date'] = df1['Transaction Date'].replace(['ERROR', 'UNKNOWN'], np.nan)
df1[['Payment Method']]=df1[['Payment Method']].fillna("Unknown")
df1[['Location']]=df1[['Location']].fillna("Unknown")
df1[['Transaction Date']]=df1[['Transaction Date']].fillna("Unknown")

# Replace ERROR and UNKNOWN with NaN
#df1['Transaction Date'] = df1['Transaction Date'].replace(['ERROR', 'UNKNOWN'], np.nan)

# Convert to datetime format
df1['Transaction Date']=pd.to_datetime(df['Transaction Date'],errors="coerce")

# Fills in the date with forward or before fill
df1["Transaction Date"] = df1["Transaction Date"].ffill().bfill()

# Make sure price is numeric
df1["Price Per Unit"] = pd.to_numeric(df1["Price Per Unit"], errors="coerce")

# Remove bad prices
valid_prices = df1[(df1["Item"].notna()) & (df1["Price Per Unit"] > 0)]

# Typical price per item
price_lookup = valid_prices.groupby("Item")["Price Per Unit"].median()

def guess_item(row):
    if pd.isna(row["Item"]) and row["Price Per Unit"] > 0:
        # Compare this price to known item prices
        diffs = abs(price_lookup - row["Price Per Unit"])
        return diffs.idxmin()  # item with closest price
    return row["Item"]

df1["Item"] = df1.apply(guess_item, axis=1)

#Dropping Items that still don't have a value in them
df1=df1.dropna(subset=["Item"])

#Prints a CSV to test
df1.to_csv('Finished Cafe Dataset6.csv', index=False)

#Printing the dataset in python vscode
print(df1)
print(df1.info)
print(df1.isnull().sum())




