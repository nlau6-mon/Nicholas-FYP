import numpy as np
import pandas as pd
import scipy.io
import pickle
from tempfile import TemporaryFile
from scipy.io import loadmat
from numpy import asarray
from numpy import save

#My Raw Tray Data (510 Bands)
raw_tray_data = np.load('Tray_Data_Raw.npy', allow_pickle=True)

#Matlab Data (246 Bands)
annots = loadmat('0013_Data.mat') 
wavelengths = annots['wavelengths']

#Creating Arrays
Value_Array =  np.linspace(0, 509, 510, 1)
Raster_Array = np.linspace(444.00, 2480.00, 510, 1)
Raster_Array = np.expand_dims(Raster_Array, axis=1)

#The Size of a Reduced Wavelength Band Tray
Minimized_Tray = np.empty([1268, 28, 246])

New_Raster_Array = []
## Iteration Starts Here
## Will need to iterate 246 Times (Up to index 245)

j = 0
k = 0
for j in range(1268):
    for k in range(28):
        for i in range (246):
            Selection_Bound_Array = Raster_Array - wavelengths[i]
            index_location_upper = list(Selection_Bound_Array).index(min(Selection_Bound_Array[Selection_Bound_Array>0]))
            index_location_lower = index_location_upper - 1
            Higher_Weight = (wavelengths[i] - Raster_Array[index_location_lower])/4
            Lower_Weight = (Raster_Array[index_location_upper] - wavelengths[i])/4
            Array_Value = Higher_Weight*raw_tray_data[j,k,index_location_upper] + Lower_Weight*raw_tray_data[j,k,index_location_lower]
            Minimized_Tray[j,k,i] = Array_Value
        

save('Reduced_Tray_Data.npy', Minimized_Tray)