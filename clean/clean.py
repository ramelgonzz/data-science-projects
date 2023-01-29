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

#create 2 dataframes and combine them together
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                   'B': ['B0', 'B1', 'B2', 'B3'],
                   'C': ['C0', 'C1', 'C2', 'C3'],
                   'D': ['D0', 'D1', 'D2', 'D3']},
                  index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                   'B': ['B4', 'B5', 'B6', 'B7'],
                   'C': ['C4', 'C5', 'C6', 'C7'],
                   'D': ['D4', 'D5', 'D2', 'D7']},
                  index=[4, 5, 6, 7])

# Use the concat() function to combine the dataframes along axis=0 (rows)
result = pd.concat([df1, df2], axis=0)

# Print the resulting dataframe
print(result)