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
#%%
drop_features = ['First Column', 'Second Column', 'Third Column', 'Fourth Column', 'Fifth Column', 'Sixth Column', 'Seventh Column']
df.drop(drop_features, inplace=True, axis=1)

df['Seventh Column'] = pd.to_numeric(extr)
df['Seventh Column'].dtype #convert to float
df.to_csv('output.csv')

#...