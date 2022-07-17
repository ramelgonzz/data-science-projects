#imports and exploratory analysis
#%%
import io
import numpy as np
import pandas as pd
#%%

df = pd.read_csv("posts.csv")
df.head()

df.drop(columns=['Description', 'Comment', 'Modified'])
df.head()
#%%
df.to_csv('output.csv')