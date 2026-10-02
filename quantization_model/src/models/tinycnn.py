import torch
import torch.nn as nn
import torch.ao.quantization as tq


class TinyCNN(nn.Module):
    """TinyCNN architecture ported from Edge_v8.ipynb."""

    def __init__(self, num_classes: int = 10, enable_quant_stubs: bool = False):
        super().__init__()
        self.quant = tq.QuantStub() if enable_quant_stubs else nn.Identity()
        self.dequant = tq.DeQuantStub() if enable_quant_stubs else nn.Identity()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        tq.fuse_modules(
            self,
            [
                ["conv1", "bn1", "relu1"],
                ["conv2", "bn2", "relu2"],
                ["conv3", "bn3", "relu3"],
                ["fc1", "relu4"],
            ],
            inplace=True,
        )
