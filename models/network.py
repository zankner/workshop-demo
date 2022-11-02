import torch
import torch.nn as nn

import torchvision


class LeNet(nn.Module):

    def __init__(self, layer_1_w, layer_2_w):
        super().__init__()

        # Model weights
        self.layer_1 = nn.Linear(in_features=28*28, out_features=layer_1_w)
        self.layer_2 = nn.Linear(in_features=layer_1_w, out_features=layer_2_w)
        self.layer_3 = nn.Linear(in_features=layer_2_w, out_features=10)

        # Activations
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Input is 28 x 28 so we flatten
        x = x.view(-1, 28*28)

        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.layer_3(x)
        return self.softmax(x)


if __name__ == "__main__":
    model = LeNet(64, 32)
    x = torch.rand(5, 28, 28)
    model(x)

