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

new_train_source = np.load('Reduced_Tray_Data.npy', allow_pickle=True)
my_data_standardized_array = [];

for i in range(56):
    for k in range (28):
        my_data = new_train_source[348 + i,k, :]
        my_data_mean = statistics.mean(my_data)
        my_data_stdev = statistics.stdev(my_data)
        my_data_standardized = (my_data - my_data_mean)/my_data_stdev
        my_data_standardized_array.append(my_data_standardized)

my_data_standardized_array = np.array(my_data_standardized_array)
save('NewTest_Standardized_Tile.npy', my_data_standardized_array)     