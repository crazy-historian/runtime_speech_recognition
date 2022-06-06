import torch.nn as nn

VGG16_A = [64, 'MaxPool', 128, 'MaxPool', 256, 256, 'MaxPool', 512, 512, 'MaxPool', 512, 512,
           'MaxPool']

VGG16_C = [64, 64, 'MaxPool', 128, 128, 'MaxPool', 256, 256, 256, 'MaxPool', 512, 512, 512, 'MaxPool', 512, 512, 512,
           'MaxPool']

VGG16_E = [64, 64, 'MaxPool', 128, 128, 'MaxPool', 256, 256, 256, 256, 'MaxPool', 512, 512, 512, 512, 'MaxPool', 512,
           512, 512, 512,
           'MaxPool']


class VGG16(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, architecture: list):
        super().__init__()
        self.in_channels = in_channels
        self.conv_layers = self._create_conv_layers(architecture)
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x

    def _create_conv_layers(self, architecture: list):
        layers = list()
        in_channels = self.in_channels
        for l in architecture:
            if isinstance(l, int):
                out_channels = l
                layers.extend([nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                               nn.BatchNorm2d(l),
                               nn.ReLU()])
                in_channels = l
            elif l == 'MaxPool':
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))

        return nn.Sequential(*layers)
