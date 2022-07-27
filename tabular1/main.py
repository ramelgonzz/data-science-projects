#data cleaning/manipulation project
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
#%%
df = pd.read_csv("olympics.csv")
df.head()
#%%
df.tail()
#%%
df.index
#%%
df.columns
#%%
all_countries = df.loc[0:146, ['0','15']]
all_countries.fillna('', inplace=True)
all_countries.head()
#%%
all_countries.tail()
#%%
fig = px.histogram(all_countries, x='0', y='15', title='aggregated olympic medals per country', labels={'0':'countries','15':'totals'})
fig.show()
# %%
