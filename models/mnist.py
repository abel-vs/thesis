import torch.nn as nn
import torch.nn.functional as F

# Basic Mnist model existing of two convolutional layers and two fully connected layers.
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1) # [1, 28, 28] -> [20, 24, 24]
        self.conv2 = nn.Conv2d(20, 50, 5, 1) # [20, 12, 12] -> [50, 8, 8]
        self.fc1 = nn.Linear(4*4*50, 500) # [50, 4, 4] -> [500]
        self.fc2 = nn.Linear(500, 10) # [500] -> [10]

    def forward(self, x):
        x = F.relu(self.conv1(x)) # [1, 28, 28] -> [20, 24, 24]
        x = F.max_pool2d(x, 2, 2) # [20, 24, 24] -> [20, 12, 12]
        x = F.relu(self.conv2(x)) # [20, 12, 12] -> [50, 8, 8]
        x = F.max_pool2d(x, 2, 2)  # [50, 8, 8] -> [50, 4, 4]
        x = x.view(-1, 4*4*50) # [50, 4, 4] -> [50*4*4]
        x = F.relu(self.fc1(x)) # [50*4*4] -> [500]
        x = self.fc2(x) # [500] -> [10]
        return F.log_softmax(x, dim=1) # [10]

if __name__ == '__main__':
    model = MnistModel()
    print(list(model.modules())[0])


# Mnist model that uses only one convolutional layer
class MnistStudent(nn.Module):
    def __init__(self):
        super(MnistStudent, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1) # [1, 28, 28] -> [20, 24, 24]
        self.fc1 = nn.Linear(12*12*20, 500) # [20, 12, 12] -> [500]
        self.fc2 = nn.Linear(500, 10) # [500] -> [10]

    def forward(self, x):
        x = F.relu(self.conv1(x)) # [1, 28, 28] -> [20, 24, 24]
        x = F.max_pool2d(x, 2, 2) # [20, 24, 24] -> [20, 12, 12]
        x = x.view(-1, 12*12*20) # [20, 12, 12] -> [20*12*12]
        x = F.relu(self.fc1(x)) # [20*12*12] -> [500]
        x = self.fc2(x) # [500] -> [10]
        return F.log_softmax(x, dim=1) # [10]

if __name__ == '__main__':
    model = MnistStudent()
    print("Modules:", list(model.modules())[0])


# Mnist model that uses only two linear layers
class MnistSmallLinear(nn.Module):
    def __init__(self):
        super(MnistSmallLinear, self).__init__()
        self.linear_1 = nn.Linear(784, 50)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(50, 10)

    def forward(self, input):
        scores = self.linear_1(input)
        scores = self.relu(scores)
        scores = self.linear_2(scores)
        return scores