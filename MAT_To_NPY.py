import numpy as np
import pandas as pd
import scipy.io
import pickle
from tempfile import TemporaryFile
from scipy.io import loadmat
from numpy import asarray
from numpy import save

annots = loadmat('0016_Data.mat')    ####CHANGE
wavelengths = annots['wavelengths']
data = annots['Data_Array']

labels = np.empty([data.shape[0], 1], dtype="S17")
labels[0:data.shape[0], 0] = "Glaucophane"   ####CHANGE

save('0016_Data.npy', data)     ####CHANGE
save('0016_Label.npy', labels)  ####CHANGE