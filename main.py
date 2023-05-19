import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.relu(x)


net = Net()
alpha = 0.01  # check valid from this value
optimizer = optim.SGD(net.parameters(), lr=alpha, momentum=0.9)

criterion = nn.NLLLoss()
