#imports and exploratory analysis
#%%
import io
import numpy as np
import pandas as pd
#%%

df = pd.read_csv("posts.csv")
df.head()

df.dropna(thresh=3) #remove rows containing less than 3 elements
df.interpolate(method='linear', order=5) #Fill NaN values using linear interpolation
df.head()
# df.dropna(subset=['Date'], inplace = True) Remove rows with a NULL value in the "Date" column
# x = df["First Column"].mean() Calculate the MEAN, and replace any empty values with it
# df["First Column"].fillna(x, inplace = True) 
#%%
drop_features = ['First Column', 'Second Column', 'Third Column', 'Fourth Column', 'Fifth Column', 'Sixth Column', 'Seventh Column']
df.drop(drop_features, inplace=True, axis=1)

df['Seventh Column'] = pd.to_numeric(extr)
df['Seventh Column'].dtype #convert to float
df.to_csv('output.csv')

df.drop_duplicates() #remove duplicate rows
df["calories"].replace(-1, None, inplace=True) #replace invalid values with a another value
df.rename(columns={"old_name": "new_name"}, inplace=True) #rename columns
df.drop(["column_name"], axis=1, inplace=True) #remove columns
df.to_csv('output2.csv') #save to second file