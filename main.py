import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import FloatTensor

import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


net = Net()
alpha = 0.01  # check valid from this value
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=alpha, momentum=0.9)

# criterion = nn.NLLLoss()

X_tr = [[1., 2.], [2.5, 3.], [3., 3.], [3.3, 0.5]]
Y_tr = [3., 5.5, 6., 4.]

X = FloatTensor(X_tr)
Y = FloatTensor(Y_tr).view(-1, 1)

losses = []
for epoch in range(1000):
    y_pred = net(X)

    loss = criterion(y_pred, Y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=[x for x in range(len(losses))], y=losses))

fig.show()