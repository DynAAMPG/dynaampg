import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class DynAAM(nn.Module):
    def __init__(self, num_features, num_classes, s=30.0, m=0.50):
        super(DynAAM, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label_one_hot=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label_one_hot is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)

        output = logits * (1 - label_one_hot) + target_logits * label_one_hot
        
        # one_hot = torch.zeros_like(logits)
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output