#from https://www.kaggle.com/code/tyrionlannisterlzy/xgboost-dnn-ensemble-lb-0-980
#determine the state of the participant using 60 second sequences of tabularsensor data
#%%
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode
init_notebook_mode(connected = True)
import warnings
warnings.filterwarnings('ignore')
#%%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission = pd.read_csv("sample_submission.csv")
labels = pd.read_csv("train_labels.csv")

train
# %%
train.describe()
# %%
#check for missing data
train.isnull().sum(axis=0)
# %%
labels.head()
# %%
train =train.merge(labels,how='left', on=["sequence"])
train.head()
# %%
#use a heatmap to calculate correlation
# set the size of the map
features  = [col for col in test.columns if col not in ("sequence","step","subject")]
plt.figure(figsize = (15,7))

hm = sns.heatmap(train[features].corr(),    # data
                cmap = 'coolwarm',# style
                annot = True,     # True to show the specific values
                fmt = '.2f',      # set the precision
                linewidths = 0.05)
plt.title('Correlation Heatmap for Train dataset', 
              fontsize=14, 
              fontweight='bold')
# %%
col_t=["sensor_00","sensor_01","sensor_03","sensor_06","sensor_07","sensor_09","sensor_11"]

# set the size of the map
plt.figure(figsize = (9,5))

hm = sns.heatmap(train[col_t].corr(),    # data
                cmap = 'coolwarm',      
                annot = True,     
                fmt = '.2f', 
                linewidths = 0.05)
plt.title('Correlation Heatmap for Selected columns from Train dataset', 
              fontsize=14, 
              fontweight='bold')
# %%
#sensor data
sequences = [0, 1, 2, 3, 4, 5]
figure, axes = plt.subplots(13, len(sequences), sharex=True, figsize=(16, 16))
for i, sequence in enumerate(sequences):
    for sensor in range(13):
        sensor_name = f"sensor_{sensor:02d}"
        plt.subplot(13, len(sequences), sensor * len(sequences) + i + 1)
        plt.plot(range(60), train[train.sequence == sequence][sensor_name],
                color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10])
        if sensor == 0: plt.title(f"Sequence {sequence}")
        if sequence == sequences[0]: plt.ylabel(sensor_name)
figure.tight_layout(w_pad=0.1)
plt.suptitle('Selected Time Series', y=1.02)
plt.show()
# %%
def aggregated_features(df, aggregation_cols = ['sequence'], prefix = ''):
    agg_strategy = {'sensor_00': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_01': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_02': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_03': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_04': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_05': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_06': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_07': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_08': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_09': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_10': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_11': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_12': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                   }
    group = df.groupby(aggregation_cols).aggregate(agg_strategy)
    group.columns = ['_'.join(col).strip() for col in group.columns]
    group.columns = [str(prefix) + str(col) for col in group.columns]
    group.reset_index(inplace = True)
    
    temp = (df.groupby(aggregation_cols).size().reset_index(name = str(prefix) + 'size'))
    group = pd.merge(temp, group, how = 'left', on = aggregation_cols,)
    return group
# %%
train_merge_data = aggregated_features(train, aggregation_cols = ['sequence', 'subject'])
test_merge_data = aggregated_features(test, aggregation_cols = ['sequence', 'subject'])
# %%
train_subjects_merge_data = aggregated_features(train, aggregation_cols = ['subject'], prefix = 'subject_')
test_subjects_merge_data = aggregated_features(test, aggregation_cols = ['subject'], prefix = 'subject_')
# %%
train_subjects_merge_data.head()
# %%
#lagging
train['sensor_00_lag_01'] = train['sensor_00'].shift(1)
train['sensor_00_lag_10'] = train['sensor_00'].shift(10)
train.head(15)
# %%
#merging before training
train_merge_data = train_merge_data.merge(labels, how = 'left', on = 'sequence')
# %%
train_merge_data = train_merge_data.merge(train_subjects_merge_data, how = 'left', on = 'subject')
test_merge_data = test_merge_data.merge(test_subjects_merge_data, how = 'left', on = 'subject')
train_merge_data.head()
# %%
test_merge_data.head()
# %%
#post process info for the model
ignore = ['sequence', 'state', 'subject']
features = [feat for feat in train_merge_data.columns if feat not in ignore]
target_feature = 'state'
# %%
#train: test split
from sklearn.model_selection import train_test_split
test_size_pct = 0.3
X_train, X_valid, y_train, y_valid = train_test_split(
                                train_merge_data[features], 
                                train_merge_data[target_feature], 
                                test_size = test_size_pct, 
                                random_state = 2022)
# %%
#xgboost
from xgboost  import XGBClassifier

params = {'n_estimators': 8192,
          'max_depth': 7,
          'learning_rate': 0.1,
          'subsample': 0.96,
          'colsample_bytree': 0.80,
          'reg_lambda': 1.50,
          'reg_alpha': 6.10,
          'gamma': 1.40,
          'random_state': 16,
          'objective': 'binary:logistic',
          #'tree_method': 'gpu_hist',
         }

xgb = XGBClassifier(**params)
xgb.fit(X_train, y_train, 
        eval_set = [(X_valid, y_valid)], 
        eval_metric = ['auc','logloss'], 
        early_stopping_rounds = 64, 
        verbose = 32)
# %%
