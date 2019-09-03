# model trainを入れる？


from util import read_csv
from dataloader import Dataset, Transform
from models import TwoLayerNet
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import math
from tensorboardX import SummaryWriter

LABELLEN = 5
DATASET_FILEPATH = '../create_dataset/dataset/*.csv'

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 2048, 1024, LABELLEN
epochs = 30
batch_size = 128
model_path = 'models/model_8_gestures_0903.pt'
LOG_PATH = "logs/" + '0903_lr0.1-1'
writer = SummaryWriter(log_dir=LOG_PATH)

model = TwoLayerNet(D_in, D_out)
criterion = torch.nn.MSELoss()
lr = 1e-1
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def train(data, model, criterion, optimizer):
    model.train()
    global lr
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    data_set = Dataset(data, LABELLEN, transform=Transform())
    train_Dataset, test_Dataset = torch.utils.data.random_split(
        data_set, [train_size, test_size])

    for epoch in range(epochs):
        dataloader = torch.utils.data.DataLoader(
            train_Dataset, batch_size=batch_size, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(
            test_Dataset, batch_size=batch_size, shuffle=True)
        for i, d in enumerate(dataloader):
            [input, label] = d
            # y_pred = model(input.float())
            y_pred = model(input.to(device).float())
            loss = criterion(y_pred.to(device), label.float().to(device))

            if i % 100 == 0:
                it = iter(dataloader_test)
                [y_test, y_label] = next(it)
                # print(y_test, y_label)
                y_test_pred = model(y_test.to(device).float())
                loss_test = criterion(y_test_pred.to(
                    device).float(), y_label.float().to(device))
                dif = math.sqrt(loss_test.item())
                print(
                    f'{epoch:04}/{epochs:04}, {i:04}, {loss.item():02.4f}, {loss_test.item():02.4f}, dif:{dif:02.4f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        writer.add_scalar("LOSS/loss", loss.item(), epoch)
        writer.add_scalar("LOSS/loss_test", loss_test.item(), epoch)
        writer.add_scalar("dif", dif, epoch)
        writer.add_scalar("lr", lr, epoch)

        # if loss.item() < 0.0005:
        #     lr = 1e-2
        # elif loss.item() < 0.00005:
        #     lr = 1e-3
        # else:
        #     lr = 1e-4

        # for g in optimizer.param_groups:
        #     g['lr'] = lr

        # Zero gradients, perform a backward pass, and update the weights.

    # save model
    torch.save(model.state_dict(), model_path)
    print('save model')


if __name__ == "__main__":
    print('loading...')
    data = read_csv(DATASET_FILEPATH)
    print('finish')
    print(model_path)
    train(data, model, criterion, optimizer)
