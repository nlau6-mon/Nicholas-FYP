import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pickle
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import math
import statistics
from numpy import save

x_train_source = np.load('MyFinalData_Larger_Final_Standardized.npy', allow_pickle=True)
y_train_source = np.load('MyFinalLabel_Larger_Final.npy', allow_pickle=True)
new_train_source = np.load('Reduced_Tray_Data.npy', allow_pickle=True)
y_label = y_train_source

y_train_source = np.ravel(y_train_source)
le = preprocessing.LabelEncoder()
le.fit(y_train_source)
y_train_source = le.transform(y_train_source)
y_train_source = np.expand_dims(y_train_source, axis=1)