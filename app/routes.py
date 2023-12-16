from app import app
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F


class BraitCnn(nn.Module):
    def __init__(self):
        super(BraitCnn,self).__init__()
        self.brait1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.01)
        )
        self.brait2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.01)
        )
        self.brait3 = nn.Sequential(
            nn.Linear(32 * 7 * 7, 100),
            nn.Linear(100, 26)
        )




@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"
