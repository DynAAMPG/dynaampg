import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from baseline_dataset import BaselineDataset
from config import *

class BaselineAltClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BaselineAltClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

