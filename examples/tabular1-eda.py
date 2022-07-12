#https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense
#exploratoray data analysis for kaggle's may22 tabular playground series
#Interactive Python code

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
import seaborn as sns
from cycler import cycler
from IPython.display import display
import datetime

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

plt.rcParams['axes.facecolor'] = '#0057b8' # blue
plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] +
                                         plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print()
print('Train')
display(train.head(3))

print()
print('Test')
display(test.head(3))

print('Dataframe shapes:', train.shape, test.shape)
print()

#%%
train.info()

#%%
(train.target.value_counts() / len(train)).round(2)



# %%
float_features = [f for f in train.columns if train[f].dtype == 'float64']

# Training histograms
fig, axs = plt.subplots(4, 4, figsize=(16, 16))
for f, ax in zip(float_features, axs.ravel()):
    ax.hist(train[f], density=True, bins=100)
    ax.set_title(f'Train {f}, std={train[f].std():.1f}')
plt.suptitle('Histograms of the float features', y=0.93, fontsize=20)
plt.show()

# Test histograms
# fig, axs = plt.subplots(4, 4, figsize=(16, 16))
# for f, ax in zip(float_features, axs.ravel()):
#     ax.hist(test[f], density=True, bins=100)
#     ax.set_title(f'Test {f}, std={test[f].std():.1f}')
# plt.show()
# %%
# Correlation matrix of the float features
plt.figure(figsize=(12, 12))
sns.heatmap(train[float_features + ['target']].corr(), center=0, annot=True, fmt='.2f')
plt.show()

# %%
# Plot dependence between every feature and the target
def plot_mutual_info_diagram(df, features, ncols=4, by_quantile=True, mutual_info=True,
                             title='How the target probability depends on single features'):
    def H(p):
        """Entropy of a binary random variable in nat"""
        return -np.log(p) * p - np.log(1-p) * (1-p)
                 
    nrows = (len(features) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, nrows*4), sharey=True)
    for f, ax in zip(features, axs.ravel()):
        temp = pd.DataFrame({f: df[f].values,
                             'state': df.target.values})
        temp = temp.sort_values(f)
        temp.reset_index(inplace=True)
        rolling_mean = temp.state.rolling(15000, center=True, min_periods=1).mean()
        if by_quantile:
            ax.scatter(temp.index, rolling_mean, s=2)
        else:
            ax.scatter(temp[f], rolling_mean, s=2)
        if mutual_info and by_quantile:
            ax.set_xlabel(f'{f} mi={H(temp.state.mean()) - H(rolling_mean[~rolling_mean.isna()].values).mean():.5f}')
        else:
            ax.set_xlabel(f'{f}')
    plt.suptitle(title, y=0.90, fontsize=20)
    plt.show()

plot_mutual_info_diagram(train, float_features,
                         title='How the target probability depends on the float features')

# %%
int_features = [f for f in test.columns if test[f].dtype == 'int64' and f != 'id']

# Training histograms
#fig, axs = plt.subplots(4, 4, figsize=(16, 16))
figure = plt.figure(figsize=(16, 16))
# for f, ax in zip(int_features, axs.ravel()):
for i, f in enumerate(int_features):
    plt.subplot(4, 4, i+1)
    ax = plt.gca()
    vc = train[f].value_counts()
    ax.bar(vc.index, vc)
    #ax.hist(train[f], density=False, bins=(train[f].max()-train[f].min()+1))
    ax.set_xlabel(f'Train {f}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # only integer labels
plt.suptitle('Histograms of the integer features', y=1.0, fontsize=20)
figure.tight_layout(h_pad=1.0)
plt.show()

# Test histograms
# fig, axs = plt.subplots(4, 4, figsize=(16, 16))
# for f, ax in zip(int_features, axs.ravel()):
#     ax.hist(test[f], density=True, bins=100)
#     ax.set_title(f'Test {f}, std={test[f].std():.1f}')
# plt.show()
# %%
plot_mutual_info_diagram(train, int_features,
                         title='How the target probability depends on the int features')

# %%
#verify that f_27 always has length 10:
train.f_27.str.len().min(), train.f_27.str.len().max(), test.f_27.str.len().min(), test.f_27.str.len().max()
# %%
train.f_27.value_counts()
#It is important to understand whether the f_27 strings in test are the same as in training. Unfortunately, test contains 1181880 - 741354 = 440526 strings which do not occur in training.
#Insight: We must not use this string as a categorical feature in a classifier. Otherwise, the model learns to rely on strings which never occur in the test data.

# %%
pd.concat([train, test]).f_27.value_counts()
# %%
for i in range(10):
    print(f'Position {i}')
    tg = train.groupby(train.f_27.str.get(i))
    temp = pd.DataFrame({'size': tg.size(), 'probability': tg.target.mean().round(2)})
    print(temp)
    print()
# %%
#count unique characters in the string feature
unique_characters = train.f_27.apply(lambda s: len(set(s))).rename('unique_characters')
tg = train.groupby(unique_characters)
temp = pd.DataFrame({'size': tg.size(), 'probability': tg.target.mean().round(2)})
print(temp)
# %%
for df in [train, test]:
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
# %%
plot_mutual_info_diagram(train, 
                         [f for f in train.columns if f.startswith('ch')] + ['unique_characters'],
                         title='How the target probability depends on the character features')
# %%
#top 3 feature interactions
plt.rcParams['axes.facecolor'] = 'k'
plt.figure(figsize=(11, 5))
cmap = ListedColormap(["#ffd700", "#0057b8"])
# target == 0 → yellow; target == 1 → blue

ax = plt.subplot(1, 3, 1)
ax.scatter(train['f_02'], train['f_21'], s=1,
           c=train.target, cmap=cmap)
ax.set_xlabel('f_02')
ax.set_ylabel('f_21')
ax.set_aspect('equal')
ax0 = ax

ax = plt.subplot(1, 3, 2, sharex=ax0, sharey=ax0)
ax.scatter(train['f_05'], train['f_22'], s=1,
           c=train.target, cmap=cmap)
ax.set_xlabel('f_05')
ax.set_ylabel('f_22')
ax.set_aspect('equal')

ax = plt.subplot(1, 3, 3, sharex=ax0, sharey=ax0)
ax.scatter(train['f_00'] + train['f_01'], train['f_26'], s=1,
           c=train.target, cmap=cmap)
ax.set_xlabel('f_00 + f_01')
ax.set_ylabel('f_26')
ax.set_aspect('equal')

plt.tight_layout(w_pad=1.0)
plt.savefig('three-projections.png')
plt.show()
plt.rcParams['axes.facecolor'] = '#0057b8' # blue
# %%
#create a ternary categorical feature to help the classifier find the borders
for df in [train, test]:
    df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
    df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)