import numpy as np
import pickle
from osgeo import gdal
import matplotlib
import matplotlib.pyplot as plt
from numpy import save

#Keep the File Name
filename = "JA0030_NVD008_0209_20110920160348_00.procSpecRefl.bin"

#Install the dataset and obtain the count
dataset = gdal.Open(filename, gdal.GA_ReadOnly)

#Creating the empty array to fill data
my_data = np.empty([1268, 28, 510])
i = 0

#Will create an array and then graph it with a degree of 4nm

for i in range(dataset.RasterCount):
	#print(i)
	band = dataset.GetRasterBand(i+1)
	data = band.ReadAsArray()
	my_data[:, :, i] = data
	#print("Band Type={}".format(gdal.GetDataTypeName(band.DataType))) #Maybe use this later

t = np.arange(444, 2484, 4)
save('Tray_Data_Raw.npy', my_data)

# fig, ax = plt.subplots()
# ax.plot(t, my_data)
# ax.set(xlabel='nm', ylabel='Spectral Value',
       # title='Sampleplot')
# plt.show()