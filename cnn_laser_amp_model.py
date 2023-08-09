#All Additive Skip steps
#Multipicative step
#Milestones Step Scheduler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sched import scheduler

# Defining Full Model Class and Forward Pass
#For Block A
k_a = [121, 101, 51, 31, 21] #size of convolution filters in each layer

#For Block B
k_b = [100, 19]
stride_b = [4, 2]
max_pool_size = [4]
max_pool_stride = [4]

#Data sizing
features_in = 2000
features_out = 2000

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        #Block A
        self.Aconv1 = nn.Conv1d(1, 1, k_a[0], stride = 1, padding = 'same', padding_mode = 'circular')
        self.Aconv2 = nn.Conv1d(1, 1, k_a[1], stride = 1, padding = 'same', padding_mode = 'circular')
        self.Aconv3 = nn.Conv1d(1, 1, k_a[2], stride = 1, padding = 'same', padding_mode = 'circular')
        self.Aconv4 = nn.Conv1d(1, 1, k_a[3], stride = 1, padding = 'same', padding_mode = 'circular')
        self.Aconv5 = nn.Conv1d(1, 1, k_a[4], stride = 1, padding = 'same', padding_mode = 'circular')
        
        #Block B
        self.BConv1 = nn.Conv1d(1, 1, k_b[0], stride_b[0]) #convolution, kernel size 100, stride 4, no padding
        self.BMaxPool1 = nn.MaxPool1d(max_pool_size[0], max_pool_stride[0]) #maxpooling size 4, stride 4, no padding
        self.BConv2 = nn.Conv1d(1, 1, k_b[1], stride_b[1]) #convolution, size 19 stride 2, no padding
        
        #Final Convolution for Error Correction
        self.CConv1 = nn.Conv1d(1, 1, 5, 1, padding = 'same', padding_mode = 'circular') #different from paper
        
    def forward(self, x):
        #Block A
        y_a = F.leaky_relu(self.Aconv1(x))+x  #skip step
        y_a = F.leaky_relu(self.Aconv2(y_a))+y_a
        y_a = F.leaky_relu(self.Aconv3(y_a))+y_a
        y_a = F.leaky_relu(self.Aconv4(y_a))+y_a
        y_a = F.leaky_relu(self.Aconv5(y_a))+y_a
        
        #Block B
        fb3 = F.tanh(self.BConv1(x)) #convolution 1 and tanh activation
        fb3 = self.BMaxPool1(fb3) #maxpool filter
        fb3 = F.tanh(self.BConv2(fb3)) #convolution 2 and tanh activation
        
        #Merge
        b_s = y_a.size(dim=0)
        merge = torch.zeros((b_s, 1, features_in))

        if (fb3.size(dim=0) == y_a.size(dim=0)):
            for x in range (y_a.size(dim=0)):
                single_inp = y_a[x, :, :].unsqueeze(0)
                single_filt = fb3[x, :, :].unsqueeze(0)
                
                single_conv = torch.zeros((1, 1, features_in), requires_grad = True)
                single_conv = F.conv1d(single_inp, single_filt, padding = 'same')
                
                merge[x , : , :] = single_conv*single_inp
                
        #Final Convolutional Layer for Error Correction
        y_final = self.CConv1(merge)
        y_final = torch.abs(y_final)
        
        return y_final
    
def training_loop(n_epochs, b_s, model, optimizer, loss_fn, inputs, gts):
    
    d_s = inputs.size(dim=0) #data_size%batch_size = 0

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    
    print(f"Training on device {device}.")
    
    #preparing tensors for device, if available --> GPU
    inputs = inputs.to(device=device)
    gts = gts.to(device=device)
    
    for epoch in range (1, n_epochs + 1): #loops through data n_epochs # of times
        loss_tot = 0
        
        for b_n in range (0, int(d_s/b_s)):
            
            inp = inputs[ b_n*b_s : (b_n+1)*b_s , : , : ]
            
            out = model(inp)
            
            gt = gts[ b_n*b_s : (b_n+1)*b_s , : , : ]
        
            loss = loss_fn(out, gt)
        
            #backward pass
            optimizer.zero_grad()
            loss.backward()
        
            #update weights
            optimizer.step()
            
            #calculate total loss from batch
            loss_tot += loss.item()
        
        avg_loss = loss_tot/b_s
        scheduler.step()
        
        if epoch == 1 or epoch%5 == 0: #get update from every 10 epochs
            print('Epoch {}, Training Loss {}'.format(
            epoch, loss_tot/b_s)) #calculate average loss for the batch

# Declaration of Hyperparemeters and Running Training Loop
learning_rate = 1e-2 #eventually make piecewise function
n_epochs = 5000 #subject to change based on data
b_s = 1
full_model = FullModel()
optimizer = optim.Adam(full_model.parameters(), lr = learning_rate)
scheduler =  optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1500,3500], gamma=0.1) #after plateau lr = lr*gamma
loss_fn = nn.MSELoss() #Mean Square Error Loss Function
inputs = torch.load('/sdf/scratch/convery1/simulated_data/sim_samples_in_1.pt')
gts = torch.load('/sdf/scratch/convery1/simulated_data/sim_samples_gts_1.pt')

training_loop(n_epochs, b_s, full_model, optimizer, loss_fn, inputs, gts)

torch.save(full_model.state_dict(), '/sdf/scratch/convery1/trained_models/model_1_on_1.pt')