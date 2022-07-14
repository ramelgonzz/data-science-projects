
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
import seaborn as sns
from cycler import cycler
from IPython.display import display
import datetime
import scipy.stats
import math
import random
import warnings

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import StandardScaler

import shap
import mplcyberpunk
import xgboost as xgb
from lightgbm import LGBMClassifier

import tensorflow as tf
import keras
from keras import Model
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from keras.layers import Dense, Input, InputLayer, Add, Concatenate
from keras.utils.vis_utils import plot_model

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#from https://www.kaggle.com/code/wti200/analysing-interactions-with-shap
#%%
#1.1 Perform eda on data
#interaction vs. correlation
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
plt.style.use("cyberpunk")
# runtime configuration of matplotlib
plt.rc("figure", 
    autolayout=False, 
    figsize=(20, 10),
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=20,
    titlepad=10,
)
train_raw = pd.read_csv('train.csv',index_col='id')
#create table
fig = plt.figure(figsize=(5, 2))
ax = fig.add_subplot(111)

table_vals = [[900000, 700000], [32, 31], [0, 0], [15, 15], [16, 16], [1, 0]]

# Draw table
the_table = plt.table(cellText=table_vals,
                      colWidths=[0.08]*2,
                      rowLabels=['#rows', '#columns', '%missing', "int64", "float64", "object"],
                      colLabels=['Train', 'Test'],
                      loc='center',
                      cellLoc='center',
                      rowColours =["purple"] * 6,  
                      colColours =["purple"] * 6,
                      cellColours=[['r', 'r'], ['r', 'r'], ['r', 'r'], ['r', 'r'], ['r', 'r'], ['r', 'r']]
                      )
the_table.auto_set_font_size(False)
the_table.set_fontsize(16)
the_table.scale(4, 4)
ax.grid(False)
ax.axis('tight')
ax.axis('off')
plt.show()

#%%
#1.2 Modelling for SHAP


# From https://www.kaggle.com/ambrosm/tpsmay22-eda-which-makes-sense
for i in range(10):
    train_raw[f'ch{i}'] = train_raw.f_27.str.get(i).apply(ord) - ord('A')
    train_raw["unique_characters"] = train_raw.f_27.apply(lambda s: len(set(s)))

features = [col for col in train_raw.columns if col != "target" and col !="f_27"]
X=train_raw[features]
y=train_raw["target"]
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.4, random_state = 42)

# Train model
lgbm_model =LGBMClassifier(n_estimators=5000, min_child_samples=80, random_state=1307)
lgbm_model.fit(X_train.values, y_train)
y_val_pred = lgbm_model.predict_proba(X_val.values)[:,1]
score = roc_auc_score(y_val, y_val_pred)
print(f"Validation AUC:{(score):.3f}")

#%%
#SHAP interaction values
#from https://www.kaggle.com/code/wti200/analysing-interactions-with-shap

# Using a random sample of the dataframe for better time computation
X_sampled = X_val.sample(20000, random_state=1307)

# explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_sampled)

#Get SHAP interaction values. Beware it is time consuming to calculate the interaction values.
#shap_interaction = explainer.shap_interaction_values(X_sampled)
#print(np.shape(shap_interaction))

#%%
#shap_interaction = np.savetxt("shap-interaction-20k.txt", X, delimiter=',')
#%%
loaded_arr = np.loadtxt('shap-interaction-20k.txt')
load_original_arr = loaded_arr.reshape(
    #loaded_arr.shape[0], loaded_arr.shape[1] // shap_interaction.shape[2], shap_interaction.shape[2])
    loaded_arr.shape[0], loaded_arr.shape[1] // 41, 41)

shap_interaction = load_original_arr
#%%
#absolute mean plot
# Get absolute mean of matrices
mean_shap = np.abs(shap_interaction).mean(0)
df = pd.DataFrame(mean_shap, index=X.columns, columns=X.columns)

# times off diagonal by 2
df.where(df.values == np.diagonal(df),df.values*2, inplace=True)

# display 
fig = plt.figure(figsize=(35, 20), facecolor='#002637', edgecolor='r')
ax = fig.add_subplot()
sns.heatmap(df.round(decimals=3), cmap='coolwarm', annot=True, fmt='.6g', cbar=False, ax=ax, )
ax.tick_params(axis='x', colors='w', labelsize=15, rotation=90)
ax.tick_params(axis='y', colors='w', labelsize=15)

plt.suptitle("SHAP interaction values", color="white", fontsize=60, y=0.97)
plt.yticks(rotation=0) 
plt.show()
#For instance we can see that the main effect is large for features f_21 (0.539) , f_26 (0.612) and unique_characters(0.712). This tells us that these features tend to have large positive or negative main effects. In other words, these features tend to have a significant impact on the model’s predictions.
#Similarly, we can see that interaction effects for (f_00, f_26) --> (0.513) and (f_02, f_21) --> (0.482) are significant. These are just some examples.
#%%
#feature interaction analysis
#plot feature interaction
def plot_feature_interaction(f1, f2):
    # dependence plot
    fig = plt.figure(tight_layout=True, figsize=(20,10))
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)


    ax0 = fig.add_subplot(spec[0, 0])
    minv, maxv = np.percentile(X_sampled, [1, 99])
    shap.dependence_plot(f1, shap_values[1], X_sampled, display_features=X_sampled, interaction_index=f2, ax=ax0, show=False)
    ax0.yaxis.label.set_color('white')          #setting up Y-axis label color to blue
    ax0.xaxis.label.set_color('white')          #setting up Y-axis label color to blue
    ax0.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    ax0.tick_params(axis='y', colors='white')    #setting up X-axis tick color to red
    ax0.set_title(f'SHAP main effect', fontsize=10)

    ax1 = fig.add_subplot(spec[0, 1])
    shap.dependence_plot((f1, f2), shap_interaction, X_sampled, display_features=X_sampled, ax=ax1, axis_color='w', show=False)
    ax1.yaxis.label.set_color('white')          #setting up Y-axis label color to blue
    ax1.xaxis.label.set_color('white')          #setting up Y-axis label color to blue
    ax1.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    ax1.tick_params(axis='y', colors='white')    #setting up X-axis tick color to red
    ax1.set_title(f'SHAP interaction effect', fontsize=10)


    temp = pd.DataFrame({f1: train_raw[f1].values,
                    'target': train_raw.target.values})
    temp = temp.sort_values(f1)
    temp.reset_index(inplace=True)
    
    ax3 = fig.add_subplot(spec[1, 0])
    sns.scatterplot(x=temp[f1], y=temp.target.rolling(15000, center=True).mean(), data=temp, ax=ax3, s=2)
    ax3.set_title('How the target probability depends on f_02', fontsize=10)

    temp = pd.DataFrame({f1: train_raw.loc[train_raw["f_30"]==2,f1].values,
                    'target': train_raw.loc[train_raw["f_30"]==2,'target'].values})
    temp = temp.sort_values(f1)
    temp.reset_index(inplace=True)
    
    ax4 = fig.add_subplot(spec[1, 1])
    sns.scatterplot(x=temp[f1], y=temp.target.rolling(15000, center=True).mean(), data=temp, ax=ax4, s=2)
    ax4.set_title('How the target probability depends on f_24 & f_30==2', fontsize=10)


    temp = pd.DataFrame({f1: train_raw.loc[train_raw["f_30"]!=2,f1].values,
                    'target': train_raw.loc[train_raw["f_30"]!=2,'target'].values})
    temp = temp.sort_values(f1)
    temp.reset_index(inplace=True)
    
    ax5 = fig.add_subplot(spec[1, 2])
    sns.scatterplot(x=temp[f1], y=temp.target.rolling(15000, center=True).mean(), data=temp, ax=ax5, s=2)
    ax5.set_title('How the target probability depends on f_24 & f_30!=2', fontsize=10)
    
    plt.suptitle("Feature Interaction Analysis\n f_24 and f_30", fontsize=30, y=1.15)
    fig.tight_layout()

    plt.show()

plt.style.use("cyberpunk")

f1='f_24'
f2='f_30'
plot_feature_interaction(f1, f2)
#From the first plot one can conclude that for f_30 == 2 main SHAP effect of f_24 increases for the predictions as f_24 gets larger.
#From the second plot one can conlcude that for f_30 == 2 the interaction effect between f_30 & f_24 decreases as f_24 get larger. The opposite is true for f_30 != 2.
#The third plot indicates that as f_24 gets larger the probability for state==1 increases(see AmbrosM). The fourth and fifth plot are the same as third plot but they are conditioned on f_30==2 and f_30!=2. The results are clearly different.
#%%
#More on local effect levels
#plot function
plt.style.use("cyberpunk")

def plot_feature_interaction(f1, f2):
    # dependence plot
    fig = plt.figure(tight_layout=True, figsize=(20,10))
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)


    ax0 = fig.add_subplot(spec[0, 0])
    shap.dependence_plot(f1, shap_values[1], X_sampled, display_features=X_sampled, interaction_index=None, ax=ax0, show=False)
    ax0.yaxis.label.set_color('white')          #setting up Y-axis label color to blue
    ax0.xaxis.label.set_color('white')          #setting up Y-axis label color to blue
    ax0.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    ax0.tick_params(axis='y', colors='white')    #setting up X-axis tick color to red
    ax0.set_title(f'SHAP main effect', fontsize=10)

    ax1 = fig.add_subplot(spec[0, 1])
    shap.dependence_plot((f1, f2), shap_interaction, X_sampled, display_features=X_sampled, ax=ax1, axis_color='w', show=False)
    ax1.yaxis.label.set_color('white')          #setting up Y-axis label color to blue
    ax1.xaxis.label.set_color('white')          #setting up Y-axis label color to blue
    ax1.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    ax1.tick_params(axis='y', colors='white')    #setting up X-axis tick color to red
    ax1.set_title(f'SHAP interaction effect', fontsize=10)

    ax2 = fig.add_subplot(spec[1, 0])
    sns.scatterplot(x=f1, y=f2, data=train_raw, hue="target", ax=ax2, s=2)
    ax2.text(-1.5, -5, "1", fontsize=18, verticalalignment='top', rotation="horizontal", color="k", fontproperties="smallcaps")
    ax2.text(0, 1, "2", fontsize=18, verticalalignment='top', rotation="horizontal", color="k", fontproperties="smallcaps")
    ax2.text(1, 7, "3", fontsize=18, verticalalignment='top', rotation="horizontal", color="k", fontproperties="smallcaps")

    ax2.set_title(f'scatter plot', fontsize=10)

    temp = pd.DataFrame({f1: train_raw[f1].values,'target': train_raw.target.values})
    temp = temp.sort_values(f1)
    temp.reset_index(inplace=True)
    
    ax3 = fig.add_subplot(spec[1, 1])
    sns.scatterplot(x=temp[f1], y=temp.target.rolling(15000, center=True).mean(), data=temp, ax=ax3, s=2)
    ax3.set_title('How the target probability depends on f_02', fontsize=10)
    
    plt.suptitle("Feature Interaction Analysis\n f_02 and f_21", fontsize=30, y=1.15)
    plt.show()

f1='f_02'
f2='f_21'
plot_feature_interaction(f1, f2)
#From the first plot one can conclude that as f_02 increases SHAP effect gets larger on the predictions.
#The third plot is very interesting. We can clearly see that there is no correlation between f_02 and f_21 but there is an interaction with the target. Based on the target we can divide the plot in to 3 different regions.


#####MODELING#####

#from https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model
#modeling using LightGBM

#The unique_characters feature will be used for the f_27 column.

#short eda
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplcyberpunk

sns.set_style('darkgrid')
plt.style.use("cyberpunk")

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

#%%
display(train_df.head())
display(test_df.head())

#%%
print(train_df.shape)
print(test_df.shape)
train_df.info()

#%%
def val_count_df(df, column_name, sort=True):
    value_count = df[column_name].value_counts(sort=sort).reset_index().rename(columns={column_name:"Value Count","index":column_name}).set_index(column_name)
    value_count["Percentage"] = df[column_name].value_counts(sort=sort,normalize=True)*100
    value_count = value_count.reset_index()
    return value_count

#%%
target_count = val_count_df(train_df, "target")
display(target_count)
target_count.set_index("target").plot.pie(y="Value Count", figsize=(10,7), legend=False, ylabel="");

#%%
feature_cols = [col for col in train_df.columns if "f_" in col]
dtype_cols = [train_df[i].dtype for i in feature_cols]
dtypes = pd.DataFrame({"features":feature_cols, "dtype":dtype_cols})
float_cols = dtypes.loc[dtypes["dtype"] == "float64", "features"].values.tolist()
int_cols = dtypes.loc[dtypes["dtype"] == "int64", "features"].values.tolist()

#%%
plt.subplots(figsize=(25,20))
sns.heatmap(train_df.corr(),annot=True, cmap="YlGnBu", fmt = '0.2f', vmin=-1, vmax=1, cbar=False);

#%%
plt.subplots(figsize=(25,35))
for i, column in enumerate(float_cols):
    plt.subplot(6,3,i+1)
    sns.histplot(data=train_df, x=column, hue="target")
    plt.title(column)

#%%
plt.subplots(figsize=(25,30))
for i, column in enumerate(int_cols):
    val_count = train_df[column].value_counts()
    ax = plt.subplot(5,3,i+1)
    #sns.barplot(x=val_count.index,y=val_count.values)
    ax.bar(val_count.index, val_count.values)
    ax.set_xticks(val_count.index)
    plt.title(column)

#%%
import string
alphabet_upper = list(string.ascii_uppercase)

char_counts = []
for character in alphabet_upper:
    char_counts.append(train_df["f_27"].str.count(character).sum())
char_counts_df = pd.DataFrame({"Character": alphabet_upper, "Character Count": char_counts})
char_counts_df = char_counts_df.loc[char_counts_df["Character Count"] > 0]
print(np.sum(char_counts)) #No other hidden characters

plt.subplots(figsize=(20,7))
sns.barplot(data = char_counts_df, x="Character", y="Character Count", color="cyan");
plt.title("Total number of characters in f_27 - train");

#%%
char_counts_df = char_counts_df.set_index("Character", drop=False)
for i in range(10):
    char_counts_df["character"+str(i+1)] = train_df["f_27"].str[i].value_counts()
char_counts_df = char_counts_df.fillna(0)


f,ax = plt.subplots(figsize=(20,30))
character_cols = [i for i in char_counts_df.columns if "character" in i]
for i, column in enumerate(character_cols):
    ax = plt.subplot(5,2,i+1)
    ax = sns.barplot(data = char_counts_df, x="Character", y=column, color="cyan");
    plt.title("Character value counts in position: " +str(i+1));
    ax.set_ylabel("Character Count")

# %%
#feature engineering
#using the f_27 column to create new features

#f_27 has 10 character rows, create a feature per character position
#encoding the characters ordinally

#%%
display(train_df["f_27"].head(5)) #f_27 column
display(train_df["f_27"].str[0].head(5)) # character is position 1
display(train_df["f_27"].str[0].apply(lambda x: ord(x) - ord("A")).head(5)) # Encode the characters ordinally

# %%
#number of unique characters
display(train_df["f_27"].head(5)) #f_27 column
train_df["f_27"].apply(lambda x: len(set(x))).head(5) #number of unique characters
# %%
#important interaction features shown by AmbrosM, inspired by wti200
f,ax = plt.subplots(figsize=(20,20))

plt.subplot(2,2,1)
sns.scatterplot(data = train_df, x="f_00", y="f_26", hue="target", s=2);
plt.subplot(2,2,2)
sns.scatterplot(data = train_df, x="f_01", y="f_26", hue="target", s=2);
plt.subplot(2,2,3)
sns.scatterplot(data = train_df, x="f_02", y="f_21", hue="target", s=2);
plt.subplot(2,2,4)
sns.scatterplot(data = train_df, x="f_05", y="f_22", hue="target", s=2);
# %%
#adding f_00 and f_01 and then plotting against f_26
train_df["f_00 + f_01"] =  train_df["f_00"] + train_df["f_01"]
f,ax = plt.subplots(figsize=(10,10))
sns.scatterplot(data = train_df, x="f_00 + f_01", y="f_26", hue="target", s=2);
#this shows 3 distinct regions
# %%
#adding all involved features together and plotting them against a random number drawn from a normal distribution
train_df["f_00 + f_01 + f_26"] = train_df["f_00"] + train_df["f_01"] + train_df["f_26"]
train_df["f_02 + f_21"] = train_df["f_02"] + train_df["f_21"]
train_df["f_05 + f_22"] = train_df["f_05"] + train_df["f_22"]
train_df["random"] = np.random.randn(len(train_df))
# %%
f,ax = plt.subplots(figsize=(20,20))

plt.subplot(2,2,1)
sns.scatterplot(data = train_df, y="f_00 + f_01 + f_26", x="random", hue="target", s=2);
plt.subplot(2,2,2)
sns.scatterplot(data = train_df, y="f_02 + f_21", x="random", hue="target", s=2);
plt.subplot(2,2,3)
sns.scatterplot(data = train_df, y="f_05 + f_22", x="random", hue="target", s=2);

# %%
#feature engineering cont.
def feature_engineer(df):
    new_df = df.copy()
    
    # Interaction features from AmbrosM https://www.kaggle.com/code/ambrosm/tpsmay22-advanced-keras/notebook
    # Inspired by wti200 https://www.kaggle.com/code/wti200/analysing-interactions-with-shap
    new_df['i_02_21'] = (df.f_21 + df.f_02 > 5.2).astype(int) - (df.f_21 + df.f_02 < -5.3).astype(int)
    new_df['i_05_22'] = (df.f_22 + df.f_05 > 5.1).astype(int) - (df.f_22 + df.f_05 < -5.4).astype(int)
    
    i_00_01_26 = df.f_00 + df.f_01 + df.f_26
    new_df['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
    
    #Good features
    for i in range(10):
        new_df["f_27_"+str(i)] = new_df["f_27"].str[i].apply(lambda x: ord(x) - ord("A"))
    
    #good feature:
    new_df["unique_characters"] = new_df["f_27"].apply(lambda x: len(set(x)))
    
    new_df = new_df.drop(columns=["f_27", "id"])
    return new_df
# %%
%%time
train_df.drop(columns = ["f_00 + f_01", "f_00 + f_01 + f_26", "f_02 + f_21", "f_05 + f_22", "random"], inplace=True) # drop the features we made earlier for demonstration
train_df = feature_engineer(train_df)
test_df = feature_engineer(test_df)

# %%
train_df["unique_characters"].value_counts()
# %%
#Model cont.

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

#%%
y = train_df["target"]
X = train_df.drop(columns=["target"])
X_test = test_df
X.head(2)

# %%
model = LGBMClassifier(n_estimators = 10000, learning_rate = 0.1, random_state=0, min_child_samples=90, num_leaves=150, max_bins=511, n_jobs=-1)
# %%
#variation in roc_auc score across folds is very small - so we save time and use 5-fold cross validation but only evaluate 2 of the 5 folds.
#Cross-validation
def k_fold_cv(model,X,y):
    kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state = 0)

    feature_imp, y_pred_list, y_true_list, acc_list, roc_list  = [],[],[],[],[]
    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        if fold < 2: # only evaluate 2/5 folds to save time
            print("==fold==", fold)
            X_train = X.loc[train_index]
            X_val = X.loc[val_index]

            y_train = y.loc[train_index]
            y_val = y.loc[val_index]

            model.fit(X_train,y_train)

            y_pred = model.predict_proba(X_val)[:,1]

            y_pred_list = np.append(y_pred_list, y_pred)
            y_true_list = np.append(y_true_list, y_val)

            roc_list.append(roc_auc_score(y_val,y_pred))
            acc_list.append(accuracy_score(y_pred.round(), y_val))
            print("roc auc", roc_auc_score(y_val,y_pred))
            print('Acc', accuracy_score(y_pred.round(), y_val))

            try:
                feature_imp.append(model.feature_importances_)
            except AttributeError: # if model does not have .feature_importances_ attribute
                pass # returns empty list
    return feature_imp, y_pred_list, y_true_list, acc_list, roc_list, X_val, y_val
# %%
%%time
feature_imp, y_pred_list, y_true_list, acc_list, roc_list, X_val, y_val = k_fold_cv(model=model,X=X,y=y)
# %%
print("Mean accuracy Score:", np.mean(acc_list))
print("Mean ROC AUC Score:", np.mean(roc_list))
# %%
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
def plot_cm(preds,true,ax=None):
    cm = confusion_matrix(preds.round(), true)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)#display_labels 
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format = '.6g')
    plt.grid(False)
    return disp

plot_cm(y_pred_list, y_true_list);

# %%
val_preds = pd.DataFrame({"pred_prob=1":y_pred_list, "y_val":y_true_list})
f,ax = plt.subplots(figsize=(20,20))
plt.subplot(2,1,1)
ax = sns.histplot(data=val_preds, x="pred_prob=1", hue="y_val", multiple="stack", bins = 100)
#Same plot "zoomed in"
plt.subplot(2,1,2)
ax = sns.histplot(data=val_preds, x="pred_prob=1", hue="y_val", multiple="stack", bins = 100)
ax.set_ylim([0,1000]);
# %%
#feature importance
def fold_feature_importances(model_importances, column_names, model_name, n_folds = 5, ax=None, boxplot=False):
    importances_df = pd.DataFrame({"feature_cols": column_names, "importances_fold_0": model_importances[0]})
    for i in range(1,n_folds):
        importances_df["importances_fold_"+str(i)] = model_importances[i]
    importances_df["importances_fold_median"] = importances_df.drop(columns=["feature_cols"]).median(axis=1)
    importances_df = importances_df.sort_values(by="importances_fold_median", ascending=False)
    if ax == None:
        f, ax = plt.subplots(figsize=(15, 25))
    if boxplot == False:
        ax = sns.barplot(data = importances_df, x = "importances_fold_median", y="feature_cols", color="cyan")
        ax.set_xlabel("Median Feature importance across all folds");
    elif boxplot == True:
        importances_df = importances_df.drop(columns="importances_fold_median")
        importances_df = importances_df.set_index("feature_cols").stack().reset_index().rename(columns={0:"feature_importance"})
        ax = sns.boxplot(data = importances_df, y = "feature_cols", x="feature_importance", color="cyan", orient="h")
        ax.set_xlabel("Feature importance across all folds");
    plt.title(model_name)
    ax.set_ylabel("Feature Columns")
    return ax
# %%
f, ax = plt.subplots(figsize=(15, 20))
fold_feature_importances(model_importances = feature_imp, column_names = X_val.columns, model_name = "LGBM", n_folds = 2, ax=ax, boxplot=False);
# %%
def pred_test():
    pred_list = []
    for seed in range(5):
        model = LGBMClassifier(n_estimators = 10000, learning_rate = 0.1, min_child_samples=90, num_leaves=150, max_bins=511, random_state=seed, n_jobs=-1)
        model.fit(X,y)

        preds = model.predict_proba(X_test)[:,1]
        pred_list.append(preds)
    return pred_list
# %%
pred_list = pred_test()
pred_df = pd.DataFrame(pred_list).T
pred_df = pred_df.rank()
pred_df["mean"] = pred_df.mean(axis=1)
pred_df
# %%
sample_sub = pd.read_csv("sample_submission.csv")
sample_sub["target"] = pred_df["mean"]
sample_sub
# %%
