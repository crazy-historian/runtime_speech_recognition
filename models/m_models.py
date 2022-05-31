import torch.nn as nn
import torch.nn.functional as F


class M3(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=4, n_channel=256):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        #
        self.conv2 = self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class MBlock(nn.Module):
    def __init__(self, kernel_size, in_chn, out_chn, count):
        super().__init__()
        self.layers = self._init_layers(kernel_size, in_chn, out_chn, count)

    def _init_layers(self, kernel_size, in_chn, out_chn, count):
        layers = list()
        for _ in range(count):
            layers += [
                nn.Conv1d(in_channels=in_chn, out_channels=out_chn, kernel_size=kernel_size),
                nn.BatchNorm1d(out_chn),
                nn.ReLU()
            ]
            in_chn = out_chn

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class M11(nn.Module):
    def __init__(self, n_input=1, n_output=20, stride=4, n_channel=64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        ##
        self.conv_layers_2 = MBlock(kernel_size=3, in_chn=n_channel, out_chn=n_channel, count=2)
        self.pool2 = nn.MaxPool1d(4)
        self.conv_layers_3 = MBlock(kernel_size=3, in_chn=n_channel, out_chn=n_channel * 2, count=2)
        self.pool3 = nn.MaxPool1d(4)
        self.conv_layers_4 = MBlock(kernel_size=3, in_chn=n_channel * 2, out_chn=n_channel * 4, count=3)
        self.pool4 = nn.MaxPool1d(4)
        self.conv_layers_5 = MBlock(kernel_size=3, in_chn=n_channel * 4, out_chn=n_channel * 8, count=2)

        self.fc1 = nn.Linear(n_channel * 8, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv_layers_2(x)
        x = self.pool2(x)

        x = self.conv_layers_3(x)
        x = self.pool3(x)

        x = self.conv_layers_4(x)
        x = self.pool4(x)

        x = self.conv_layers_5(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=2)


class M18(nn.Module):
    def __init__(self, n_input=1, n_output=20, stride=4, n_channel=64):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        ##
        self.conv_layers_2 = MBlock(kernel_size=3, in_chn=n_channel, out_chn=n_channel, count=4)
        self.pool2 = nn.MaxPool1d(4)
        self.conv_layers_3 = MBlock(kernel_size=3, in_chn=n_channel, out_chn=n_channel * 2, count=4)
        self.pool3 = nn.MaxPool1d(4)
        self.conv_layers_4 = MBlock(kernel_size=3, in_chn=n_channel * 2, out_chn=n_channel * 4, count=4)
        self.pool4 = nn.MaxPool1d(4)
        self.conv_layers_5 = MBlock(kernel_size=3, in_chn=n_channel * 4, out_chn=n_channel * 8, count=4)

        self.fc1 = nn.Linear(n_channel * 8, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv_layers_2(x)
        x = self.pool2(x)

        x = self.conv_layers_3(x)
        x = self.pool3(x)

        x = self.conv_layers_4(x)
        x = self.pool4(x)

        x = self.conv_layers_5(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)

        return F.log_softmax(x, dim=2)
