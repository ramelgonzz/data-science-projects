#fromhttps://www.kaggle.com/code/vladvdv/simple-vanilla-pytorch-inference-gem-pooling/notebook
#%%
import pickle
import os
import gc
import cv2
import math
import copy
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import joblib
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("../timm-pytorch-image-models")
import timm
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

#%%
#config
CONFIG = {"seed": 21, # choose your lucky seed
          "img_size": 512, # training image size
          "model_name": "tf_efficientnet_b7_ns", # training model arhitecture
          "num_classes": 15587, # total individuals in training data
          "test_batch_size": 4, # choose acording to the training arhitecture and image size 
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), # gpu
          "test_mode":False, # selects just the first 200 samples from the test data, usefull for debuging purposes
          "n_fold":5,
          # ArcFace Hyperparameters
          "s": 30.0, 
          "m": 0.30,
          "ls_eps": 0.0,
          "easy_margin": False,
          "rotate_h": False,
          "public_blend": True
          }
#%%
#set seed
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])
# %%
