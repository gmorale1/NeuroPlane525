import random
import math
import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DQN_agent(nn.Module):
    def __init__(self,layer_dims):
        '''
        Builds a 4 layer deep network

        args: 
        layer_dims - list of nueron amounts at each layer

        '''
        super().__init__()
        self.model_stack = nn.Sequential(
            #4 layer network
            nn.Linear(layer_dims[0],layer_dims[1]),
            nn.ReLU(),
            nn.Linear(layer_dims[1],layer_dims[2]),
            nn.ReLU(),
            nn.Linear(layer_dims[2],layer_dims[3]),
            nn.ReLU(),
            nn.Linear(layer_dims[3],layer_dims[4]),
            nn.Tanh()
        )

    def forward(self,x):
        y_hat = self.model_stack(x)
        #layer prediction
        return y_hat 
    @staticmethod
    def init_weights(m):
 
        if isinstance(m, nn.Linear):
            # Calculate the bounds for Glorot initialization
            bound = np.sqrt(6 / (m.weight.size(0) + m.weight.size(1)))
            # Apply Glorot initialization to the weights
            torch.nn.init.uniform_(m.weight, -bound, bound)
            # Initialize biases to zeros
            m.bias.data.fill_(0.00)