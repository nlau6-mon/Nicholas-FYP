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

x_train_source = np.load('MyFinalData_Larger_Final_Standardized.npy', allow_pickle=True)
y_train_source = np.load('MyFinalLabel_Larger_Final.npy', allow_pickle=True)
new_test_data = np.load('NewTest_Standardized_Tile.npy', allow_pickle=True)

y_train_source = np.ravel(y_train_source)
le = preprocessing.LabelEncoder()
le.fit(y_train_source)
y_train_source = le.transform(y_train_source)

y_train_source = np.expand_dims(y_train_source, axis=1)

randomize = np.arange(len(y_train_source))
np.random.shuffle(randomize)
np.random.shuffle(randomize)
randomize = np.expand_dims(randomize, axis=1)

x_train_source = x_train_source[randomize]
y_train_source = y_train_source[randomize]

x_train_source = np.squeeze(x_train_source)
y_train_source = np.squeeze(y_train_source)
y_train_source = np.expand_dims(y_train_source, axis = 1)

#Sorting Files into desired data set
# Note:
# Training = 10500 Inputs
# Validation = 3000 Inputs
# Test = 500 Inputs
x_train_numpy = x_train_source[0:132299, 0:245]
y_train_numpy = y_train_source[0:132299, 0:29]
x_validate_numpy = x_train_source[132300:160650, 0:245]
y_validate_numpy = y_train_source[132300:160650, 0:29]
x_test_numpy = x_train_source[160651:189190, 0:245]
y_test_numpy = y_train_source[160651:189190, 0:29]
new_test_numpy_x = new_test_data[0:1260, 0:245]

#Converting the Train/Validate/Test Numpy Arrays into Tensor Arrays 
x_train_tensor = torch.from_numpy(x_train_numpy).float()
y_train_tensor = torch.from_numpy(y_train_numpy).float()
x_validate_tensor = torch.from_numpy(x_validate_numpy).float()
y_validate_tensor = torch.from_numpy(y_validate_numpy).float()
x_test_tensor = torch.from_numpy(x_test_numpy).float()
y_test_tensor = torch.from_numpy(y_test_numpy).float() ##Does not get used
new_test_numpy_x_tensor = torch.from_numpy(new_test_numpy_x).float()

#####Data Loader Code ################
##### NOT IN USE      ################
train_data = []
validate_data = []

for i in range(len(x_train_tensor)):
   train_data.append([x_train_tensor[i], y_train_tensor[i]])
for j in range(len(x_validate_tensor)):
    validate_data.append([x_validate_tensor[j], y_validate_tensor[j]])

trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=30)
i1, l1 = next(iter(trainloader))

validloader = torch.utils.data.DataLoader(validate_data, shuffle=True, batch_size=30)
iv1, lv1 = next(iter(validloader))
#########################################

#Defining the NN
class Net(nn.Module):
    
	#Defining the nodes and activation functions
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(245, 1024) #was 50
        self.relu4 = nn.ReLU()
        self.dout4 = nn.Dropout(0.10)
        self.fc5 = nn.Linear(1024, 1024) #was 50
        self.relu5 = nn.ReLU()
        self.dout5 = nn.Dropout(0.10)
        self.out = nn.Linear(1024, 30) #was 15 and 7
        self.out_act = nn.ReLU()
    
    #Defining the propagation path
    def forward(self, input_):
        a1 = self.fc1(input_)
        h4 = self.relu4(a1)
        dout4 = self.dout4(h4)
        a5 = self.fc5(dout4)
        h5 = self.relu5(a5)
        dout5 = self.dout5(h5)
        a5 = self.out(dout5)
        ya = self.out_act(a5)
        return ya

#Creating the Neural Network		
net = Net()
opt = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0001) #Was 3 zeros more for Learning rate Weight Decay was also 0.001
criterion = nn.CrossEntropyLoss()

#Training and Validation Stage #########################################
def train_epoch(model, opt, criterion, batch_size=50): #Was 50
    #Training Mode
    model.train()
    train_losses = []
    validate_losses = []
    for beg_i in range(0, x_train_tensor.size(0), batch_size):
        x_batch = x_train_tensor[beg_i:beg_i + batch_size, :]
        y_batch = y_train_tensor[beg_i:beg_i + batch_size, :]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        #y_batch = np.argmax(y_batch, axis=1) #Necessary for use of "CrossEntropyLoss"
        y_batch = np.squeeze(y_batch)
        #y_batch = np.expand_dims(y_batch, axis=0)
        y_batch = y_batch.long()
        # print("This is y batch")
        # print(y_batch)
        # print(y_batch.shape)

        opt.zero_grad()
        y_hat = net(x_batch)
        loss = criterion(y_hat, y_batch)
        print(loss)
        loss.backward()
        opt.step()        
        train_losses.append(loss.data.numpy())
    
	#Validation Mode
    model.eval()
    
    for beg_j in range(0, x_validate_tensor.size(0), batch_size):
        x_batch = x_validate_tensor[beg_j:beg_j + batch_size, :]
        y_batch = y_validate_tensor[beg_j:beg_j + batch_size, :]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        #y_batch = np.argmax(y_batch, axis = 1)
        y_batch = np.squeeze(y_batch)
        y_batch = y_batch.long()
        
        opt.zero_grad()
        y_hat = net(x_batch)
        loss = criterion(y_hat, y_batch)
        print("Validation")
        print(loss)
        loss.backward()
        opt.step()        
        validate_losses.append(loss.data.numpy())	
    return train_losses #Not using for the moment
	
e_losses = [] #Not using for the moment
num_epochs = 120

scheduler = optim.lr_scheduler.StepLR(opt, step_size = 30, gamma=0.5)
for e in range(num_epochs):
    e_losses += train_epoch(net, opt, criterion) #Not using the LHS for the moment
    scheduler.step()
    print(opt.param_groups[0]['lr'])

##############################################################################	



	
############################ Testing Stage ###################################
#### Below is very specific for the data that is being looked at!!! ##########
#Testing the NN after it has been trained and validated
y_check = net(x_test_tensor)

#Convert From Tensor to numpy Array
y_check = y_check.detach().numpy()

print(y_check[0:20])
print(y_test_numpy[0:20])

print(y_check[200:220])
print(y_test_numpy[200:220])

#Displaying the index of the positive classification for the test data
print("Test")
Test_Array = np.argmax(y_check, axis = 1)
print(Test_Array)
print(Test_Array.shape)

y_test_numpy_a = np.squeeze(y_test_numpy)
print(y_test_numpy_a)
print(y_test_numpy_a.shape)
# #Displaying the index of the positive classification for the test label
#print("Actual")
# Truth_Array = np.argmax(y_test_numpy, axis = 1)
# print(Truth_Array)

# print ("Total Accruacy of NN from 500 samples will be shown below:")
print (sum(Test_Array == y_test_numpy_a)/ len(Test_Array))

torch.save(net.state_dict(), 'tensor_final.pt')
##############################################################################

############################ Testing Stage 2###################################
raster_check = net(new_test_numpy_x_tensor)

#Convert From Tensor to numpy Array
raster_check = raster_check.detach().numpy()
print(raster_check)
save('Raster_Check_Later_Tile_NewLoss.npy', raster_check) 