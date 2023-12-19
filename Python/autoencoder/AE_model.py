#%% Importations

import numpy as np
import torch 
import torch.nn as nn
import torch.functional as F

#%% Autoencoders

class AE(nn.Module):
    def __init__(self,scale=2):
        super(AE, self).__init__()
        self.scale=scale
        # encoding
        self.enc = nn.Linear(scale*scale*9,int(9*scale*scale*2.5))
        # decoding
        self.dec = nn.Linear(int(9*scale*scale*2.5),scale*scale*9)

    def forward(self, x, path="all"):
        if path=="all":
            # input
            x = x.view(x.shape[0],x.shape[1], self.scale*self.scale*9)
            # encoding
            x = self.enc(x)
            x = torch.sigmoid(x)
            #decoding
            x = self.dec(x)
            x = torch.sigmoid(x)
            x = x.view(x.shape[0],x.shape[1], self.scale*3,self.scale*3)
        elif path=="encoding":
            # input
            x = x.view(x.shape[0],x.shape[1], self.scale*self.scale*9)
            # encoding
            x = self.enc(x)
            x = torch.sigmoid(x)
        elif path=="decoding":
            #decoding
            x = self.dec(x)
            x = torch.sigmoid(x)
            x = x.view(x.shape[0],x.shape[1], self.scale*3,self.scale*3)
        else :
            raise NotImplementedError
            
        return x
    
class mapping(nn.Module):
    def __init__(self,scale=2):
        super(mapping, self).__init__()
        self.scale=scale
        # mapping
        self.map = nn.Linear(int(9*scale*scale*2.5),int(9*scale*scale*2.5))

    def forward(self, x):
        # mapping

        x = self.map(x)
       
        return x
    
class CDA(nn.Module):
    def __init__(self,scale=2):
        super(CDA, self).__init__()
        self.scale=scale
        # encoding
        self.enc = nn.Linear(scale*scale*9,int(9*scale*scale*2.5))
        # mapping
        self.map = nn.Linear(int(9*scale*scale*2.5),int(9*scale*scale*2.5))
        # decoding
        self.dec = nn.Linear(int(9*scale*scale*2.5),scale*scale*9)

    def forward(self, x):
        # input
        x = x.view(x.shape[0],x.shape[1], self.scale*self.scale*9)
        # encoding
        x = self.enc(x)
        x = torch.sigmoid(x)
        # mapping
        x = self.map(x)
        #decoding
        x = self.dec(x)
        x = torch.sigmoid(x)
        x = x.view(x.shape[0],x.shape[1], self.scale*3,self.scale*3)
            
        return x
