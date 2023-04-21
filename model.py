import torch
from torch.nn import Module
from torch.nn import Sequential, Conv2d, Sigmoid, ReLU, MaxPool2d, Linear, Flatten
from torchvision import models
from dataset import *
import torch.nn as nn
torch.set_printoptions(precision=20)

class siamese(Module):
    def __init__(self):
        super(siamese, self).__init__()
        self.cnn = Sequential(
            Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2,stride=2),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2,stride=2),
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2,stride=2),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2,stride=2),
            Flatten()
        )
        self.fully_connect = Sequential(
            Linear(51200, 512), # 这里的51200是会随着h和w变的
            Linear(512,1),
            Sigmoid()
        )

    def forward (self, x1, x2):
        batch_size = x1.shape[0]
        x1 = self.cnn(x1)
        x2 = self.cnn(x2)
        ans = torch.abs(x1-x2)
        ans = ans.view(batch_size, -1)
        ans = self.fully_connect(ans)
        ans = ans.view(-1)
        return ans

if __name__ == "__main__":
    train_test = dataset(max_num=100)
    train_loader = DataLoader(train_test, batch_size=16, shuffle=False)
    network = siamese()
    for i, (first, second, label) in enumerate(train_loader):
        ans = network(first, second)
        torch.set_printoptions(precision=20)