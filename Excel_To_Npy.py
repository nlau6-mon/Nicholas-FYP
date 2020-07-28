import numpy as np
import pandas as pd
import scipy.io
import pickle
from tempfile import TemporaryFile
from scipy.io import loadmat
from numpy import asarray
from numpy import save

df = pd.read_excel('JPL_Actinolite_1.xlsx')
n_df = pd.DataFrame(df).to_numpy()
save('JPL_Actinolite_1.npy', n_df)