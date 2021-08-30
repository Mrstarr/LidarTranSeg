import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1, shortcut = None):
        super(BasicBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(in_channels, out_channels, 3, stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding = 1, bias = False),
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        res = x if self.shortcut is None else self.shortcut(x)
        return out + res


class GCN(nn.Module):

    def __init__(self, in_channels, out_channels, k_w = 3, k_h = 3):
        super(GCN, self).__init__()

        self.conv_l1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (k_w, 1),
                                 padding = 'same')
        self.conv_l2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = (1, k_h),
                                 padding = 'same')

        self.conv_r1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = (1, k_h),
                                 padding = 'same')
        self.conv_r2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = (k_w, 1),
                                 padding = 'same')

    def forward(self, x):
        x1 = self.conv_l1(x)
        x1 = self.conv_l2(x1)

        x2 = self.conv_r1(x)
        x2 = self.conv_r2(x2)

        out = x1 + x2

        return out


class ResNet34(nn.Module):

    def __init__(self, in_channels):
        super(ResNet34, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )  # out: 64 × 16 × 512
        self.layer2 = self._make_layer(64, 64, 3)  # out: 64 × 16 × 512
        self.layer3 = self._make_layer(64, 128, 4, stride = 2)  # out: 128 × 8 × 256
        self.layer4 = self._make_layer(128, 256, 6, stride = 2)  # out: 256 × 4 × 128
        self.layer5 = self._make_layer(256, 256, 3, stride = 2)  # out: 256 × 2 × 64

    def _make_layer(self, in_channels, out_channels, num_block, stride = 1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        layers = [BasicBlock(in_channels, out_channels, stride, shortcut)]

        for _ in range(1, num_block):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x


class CPTR(nn.Module):

    def __init__(self, in_channels, num_of_classes):
        super(CPTR, self).__init__()
        self.backbone = ResNet34(in_channels)
        encoder_layer = nn.TransformerEncoderLayer(d_model = 256, nhead = 8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 6)

    def forward(self, x):
        out = self.backbone(x)
        out = out.flatten(2)
        print(out.size())
        out = self.transformer_encoder(out)
        return out


if __name__ == "__main__":
    src = torch.rand(1, 3, 512, 512)
    # src = torch.rand(4, 5, 64, 2048)
    model = CPTR(3, 19)
    output = model(src)
    print(output.size())
