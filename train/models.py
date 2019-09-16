import torch
import torch.nn as nn


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, 1024)
        self.linear2 = torch.nn.Linear(1024, 1024)
        self.linear3 = torch.nn.Linear(1024, 512)
        self.linear4 = torch.nn.Linear(512, 128)
        self.linear5 = torch.nn.Linear(128, D_out)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        self.drop_layer1 = nn.Dropout(p=0.2)
        self.drop_layer2 = nn.Dropout(p=0.2)
        self.drop_layer3 = nn.Dropout(p=0.2)
        self.drop_layer4 = nn.Dropout(p=0.2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.linear1(x)  # .clamp(min=0)
        out2 = self.relu1(self.drop_layer1(out1))
        out3 = self.linear2(out2)
        out4 = self.relu2(self.drop_layer2(out3))
        out5 = self.linear3(out4)
        out6 = self.relu3(self.drop_layer3(out5))
        out7 = self.linear4(out6)
        out8 = self.relu4(self.drop_layer4(out7))
        out9 = self.linear5(out8)
        out = self.sigmoid(out9)

        return out


class ImageRNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_inputs, n_neurons, n_outputs):
        super(ImageRNN, self).__init__()

        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.basic_rnn = nn.RNN(self.n_inputs, self.n_neurons)

        self.FC = nn.Linear(self.n_neurons, self.n_outputs)

    def init_hidden(self,):
        # (num_layers, batch_size, n_neurons)
        return (torch.zeros(1, self.batch_size, self.n_neurons))

    def forward(self, X):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        X = X.permute(1, 0, 2)

        self.batch_size = X.size(1)
        self.hidden = self.init_hidden()

        # lstm_out => n_steps, batch_size, n_neurons (hidden states for each time step)
        # self.hidden => 1, batch_size, n_neurons (final state from each lstm_out)
        lstm_out, self.hidden = self.basic_rnn(X, self.hidden)
        out = self.FC(self.hidden)

        return out.view(-1, self.n_outputs)  # batch_size X n_output
