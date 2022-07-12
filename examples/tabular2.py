#https://www.kaggle.com/code/hasanbasriakcay/tpsmay22-my100-notebook-autoblendingfunc/notebook
#%%
import numpy as np
import pandas as pd
import warnings

warnings.simplefilter("ignore")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sub = pd.read_csv("sample_submission.csv")
display(train.head())
display(test.head())
display(sub.head())
# %%
#%%
def create_features(data):
    object_data_cols = [f"f_27_{i+1}" for i in range(10)]
    object_data = pd.DataFrame(data['f_27'].apply(list).tolist(), columns=object_data_cols)
    for feature in object_data_cols:
        object_data[feature] = object_data[feature].apply(ord) - ord('A')
    
    data = pd.concat([data, object_data], 1)
    data["unique_characters"] = data.f_27.apply(lambda s: len(set(s)))
    
    ## sum
    # float
    data['f_sum_2'] = (data['f_21']+data['f_22'])
    data['f_sum_3'] = (data['f_23']-data['f_20'])
    
    continuous_feat = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_19', 'f_20', 'f_21', 'f_22', 
                       'f_23', 'f_24', 'f_25', 'f_26', 'f_28']
    
    data['f_sum']  = data[continuous_feat].sum(axis=1)
    data['f_min']  = data[continuous_feat].min(axis=1)
    data['f_max']  = data[continuous_feat].max(axis=1)
    data['f_std']  = data[continuous_feat].std(axis=1)    
    data['f_mad']  = data[continuous_feat].mad(axis=1)
    data['f_mean'] = data[continuous_feat].mean(axis=1)
    data['f_kurt'] = data[continuous_feat].kurt(axis=1)
    data['f_count_pos']  = data[continuous_feat].gt(0).count(axis=1)
    
    # int
    data['f_sum_10'] = (data['f_07']-data['f_10'])
    data['f_sum_13'] = (data['f_08']-data['f_10'])
    
    # 
    data['i_02_21'] = (data.f_21 + data.f_02 > 5.2).astype(int) - (data.f_21 + data.f_02 < -5.3).astype(int)
    data['i_05_22'] = (data.f_22 + data.f_05 > 5.1).astype(int) - (data.f_22 + data.f_05 < -5.4).astype(int)
    i_00_01_26 = data.f_00 + data.f_01 + data.f_26
    data['i_00_01_26'] = (i_00_01_26 > 5.0).astype(int) - (i_00_01_26 < -5.0).astype(int)
    
    return data

#%%
#Feature Engineering

train_fe = create_features(train.copy())
test_fe = create_features(test.copy())
train_fe.head()
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_tr = pd.DataFrame(scaler.fit_transform(train_fe.drop(['id', 'f_27', "target"], 1)), columns=train_fe.drop(['id', 'f_27', "target"], 1).columns)
test_tr = pd.DataFrame(scaler.transform(test_fe.drop(['id', 'f_27'], 1)), columns=train_fe.drop(['id', 'f_27', "target"], 1).columns)
train_tr.head()

#Modeling
#%%
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr


#%%
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.python.keras.layers import Dense, Input, InputLayer, Add, Dropout
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.models import clone_model