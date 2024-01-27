import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CharbonnierLoss(nn.Module): 
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class MSELoss(nn.Module): # MSE Loss
    """MSE Loss"""
    
    def __init__(self, eps=1e-3):
        super(MSELoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(diff * diff + self.eps*self.eps)
        return loss



