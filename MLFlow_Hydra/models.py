
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms



class SimpleCNN(nn.Module):
    def __init__(self, conv_size:int):
        self.conv_size=conv_size
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, conv_size, 3, padding=1)
        self.fc1 = nn.Linear(self.conv_size * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, self.conv_size * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x





    

