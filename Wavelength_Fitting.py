import scipy.io
import pickle
from tempfile import TemporaryFile
from scipy.io import loadmat
from numpy import asarray
from numpy import save

#Matlab Data (246 Bands)
annots = loadmat('0013_Data.mat') 
wavelengths = annots['wavelengths']

#USGS_Data Source (Approx 20000 Bands)
x_train_source = np.load('USGS_Tourmaline_1.npy', allow_pickle=True)

#Loading Data into arrays for use
unfitted_wavelengths = x_train_source[:,0]
unfitted_wavelengths = np.expand_dims(unfitted_wavelengths, axis=1)
unfitted_reflectance = x_train_source[:,1]
unfitted_reflectance = np.expand_dims(unfitted_reflectance, axis=1)

#Creating the new array to fill data
Minimized_Tray = np.empty([1, 246])

#Filling the data (for 246 bands)
for i in range (246):
    Selection_Bound_Array = unfitted_wavelengths - wavelengths[i] 
    index_location_upper = list(Selection_Bound_Array).index(min(Selection_Bound_Array[Selection_Bound_Array>0]))
    index_location_lower = index_location_upper - 1
    Higher_Weight = (wavelengths[i] - unfitted_wavelengths[index_location_lower])/(unfitted_wavelengths[index_location_upper] - unfitted_wavelengths[index_location_lower]) #Need to change the 4
    Lower_Weight = (unfitted_wavelengths[index_location_upper] - wavelengths[i])/(unfitted_wavelengths[index_location_upper] - unfitted_wavelengths[index_location_lower])  #Need to change the 4
    Array_Value = Higher_Weight*unfitted_reflectance[index_location_upper] + Lower_Weight*unfitted_reflectance[index_location_lower] #Replace raw_tray_data with the absorption values of my array
    Minimized_Tray[0,i] = Array_Value
    
#Saving the filled data
save('USGS_Tourmaline_1_Fitted.npy', Minimized_Tray)