from util import read_csv
from dataloader import Dataset, Transform
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data


LABELLEN = 5

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 2048, 100, LABELLEN


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        out = self.sigmoid(y_pred)
        return out


model = TwoLayerNet(D_in, H, D_out)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
device = torch.device("cpu")
model = model.to(device)


def train(data, model, criterion, optimizer):
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    data_set = Dataset(data, LABELLEN, transform=Transform())
    train_Dataset, test_Dataset = torch.utils.data.random_split(
        data_set, [train_size, test_size])

    epochs = 100
    for epoch in range(epochs):
        dataloader = torch.utils.data.DataLoader(
            train_Dataset, batch_size=10, shuffle=True)
        for i, d in enumerate(dataloader):
            [input, label] = d
            y_pred = model(input.float())
            loss = criterion(y_pred.float(), label.float())
            # print(loss.item())
            if i % 100 == 0:
                print(i, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    print('loading...')
    data = read_csv('../create_database/database1.csv')
    print('finish')
    data = np.array(data)

    train(data, model, criterion, optimizer)
