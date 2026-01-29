import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convds=nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.6),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)#28*28
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*28*28,128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128,2),

        )
    def forward(self,x):
        x = self.convds(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x