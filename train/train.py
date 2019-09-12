from util import read_csv, calc_acc, get_filedata
from dataloader import Dataset, Transform
from models import TwoLayerNet
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import math
from tensorboardX import SummaryWriter

LABELLEN = 5
DATASET_FILEPATH = '../create_dataset/dataset/var/*.csv'
TEST_DATASET_FILEPATH = '../create_dataset/dataset/var/test/*test.csv'


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 2048+8, 1024, LABELLEN
epochs = 5
batch_size = 128
model_path = 'models/model_7_gestures_var_0912_4_dev.pt'
LOG_PATH = "logs/" + 'var0912_lr0.1-4_dev'
writer = SummaryWriter(log_dir=LOG_PATH)

model = TwoLayerNet(D_in, D_out)

# multi gpu
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = torch.nn.MSELoss()
lr = 1e-1
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def train(**pram):
    # filedata, filelen, model, criterion, optimizer
    # filedata=filedata, filelen=filelen, filedata_test=filedata_test,
    #       filelen_test=filelen_test, model=model, criterion=criterion, optimizer=optimizer
    model.train()
    global lr
    # train_size = int(0.8 * pram['filelen'])
    # test_size = filelen - train_size

    train_Dataset = Dataset(pram['filedata'], pram['filelen'],
                            LABELLEN, transform=Transform())

    test_Dataset = Dataset(pram['filedata_test'], pram['filelen_test'],
                           LABELLEN, transform=Transform())
    # train_Dataset, test_Dataset = torch.utils.data.random_split(
    #     data_set, [train_size, test_size])

    for epoch in range(epochs):
        dataloader = torch.utils.data.DataLoader(
            train_Dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        dataloader_test = torch.utils.data.DataLoader(
            test_Dataset, batch_size=batch_size, shuffle=True)
        for i, d in enumerate(dataloader):
            [input, label] = d
            # y_pred = model(input.float())

            # print('input.size()', input.size())
            # print('label.size()', label.size())
            y_pred = pram['model'](input.to(device).float())

            # print(y_pred.cpu().detach().numpy()[0],
            #       label.cpu().detach().numpy()[0])
            loss = pram['criterion'](
                y_pred.to(device), label.float().to(device))

            if i == 0:
                it = iter(dataloader_test)
                [y_test, y_label] = next(it)
                # print(y_test, y_label)
                y_test_pred = pram['model'](y_test.to(device).float())
                loss_test = pram['criterion'](y_test_pred.to(
                    device).float(), y_label.float().to(device))
                dif = math.sqrt(loss_test.item())
                acc_all, acc_each = calc_acc(y_test_pred.cpu().detach().numpy(),
                                             y_label.cpu().detach().numpy(), threshold=0.5)
                print(
                    f'{epoch:04}/{epochs:04}, {i:04}, {loss.item():02.4f}, {loss_test.item():02.4f}, dif:{dif:02.4f}, acc:{acc_all:02.4f}')
                writer.add_scalar("LOSS/loss", loss.item(), epoch)
                writer.add_scalar("LOSS/loss_test", loss_test.item(), epoch)
                writer.add_scalar("dif", dif, epoch)
                writer.add_scalar("acc_all", acc_all, epoch)
                writer.add_scalar("lr", lr, epoch)

                for label in acc_each:
                    writer.add_scalar("acc_each/" + label,
                                      acc_each[label], epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
    # data = read_csv(DATASET_FILEPATH)

    print('train dataset')
    filedata, filelen = get_filedata(DATASET_FILEPATH)

    print('test dataset')
    filedata_test, filelen_test = get_filedata(TEST_DATASET_FILEPATH)

    print('finish')
    print(model_path)
    train(filedata=filedata, filelen=filelen, filedata_test=filedata_test,
          filelen_test=filelen_test, model=model, criterion=criterion, optimizer=optimizer)
