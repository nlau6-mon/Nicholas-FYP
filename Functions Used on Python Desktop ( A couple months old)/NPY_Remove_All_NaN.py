#Declaring Libraries
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
from numpy import save

a = []
b = []

#Arsenopyrite
x_train_source_f = np.load('0087_Data_A.npy', allow_pickle=True)
y_train_source_f = np.load('0087_Label_A.npy', allow_pickle=True)
x_train_source_g = np.load('0087_Data_B.npy', allow_pickle=True)
y_train_source_g = np.load('0087_Label_B.npy', allow_pickle=True)
x_train_arsen = np.concatenate((x_train_source_f, x_train_source_g) , axis=0)
y_train_arsen = np.concatenate((y_train_source_f, y_train_source_g) , axis=0)

for i in range(x_train_arsen.shape[0]):
    my_sum = sum(~np.isnan(x_train_arsen[i, :]))
    
    if my_sum == 256:
        a.append(x_train_arsen[i, :])
        b.append(y_train_arsen[i, :])
        print(i)

x_train_arsen = np.array(a)
y_train_arsen = np.array(b)