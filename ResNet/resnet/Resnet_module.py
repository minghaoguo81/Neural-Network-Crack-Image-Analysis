import torch
import torch.nn.functional as F

# 检查是否可以利用GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')

# resnet模块

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.skip = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels))

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(out_channels))

    def forward(self, x):
        out = self.block(x)
        identity = self.skip(x)
        out += identity
        out = F.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.cnn_layers = torch.nn.Sequential(
            ResidualBlock(3, 32, 1),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64, 2),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 256, 2),
        )

        self.type_fc = torch.nn.Sequential(
            torch.nn.Linear(256 * 4 * 4, 128),
            torch.nn.Linear(128, 64),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16,2)

        )

    def forward(self, x):
        # 向前推导
        x = self.cnn_layers(x)
        # 改变维度
        out = x.view(-1, 4 * 256 * 4)
        # 输出加上softmax
        out_type =torch.nn.functional.softmax(self.type_fc(out))
        return out_type