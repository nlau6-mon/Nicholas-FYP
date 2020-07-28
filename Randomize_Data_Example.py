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

randomize = np.arange(len(y_train_source))
np.random.shuffle(randomize)
np.random.shuffle(randomize)
randomize = np.expand_dims(randomize, axis=1)

x_train_source = x_train_source[randomize]
y_train_source = y_train_source[randomize]

x_train_source = np.squeeze(x_train_source)
y_train_source = np.squeeze(y_train_source)
y_train_source = np.expand_dims(y_train_source, axis = 1)