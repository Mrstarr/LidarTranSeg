import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, bn_d = 0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size = 1,
                               stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(planes[0], momentum = bn_d)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size = 3,
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes[1], momentum = bn_d)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


# ******************************************************************************

class LidarDecoder(nn.Module):
    """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self, bn_d = 0.01, drop_prob = 0.01, OS = 32, feature_depth = 1024):
        super(LidarDecoder, self).__init__()
        self.backbone_OS = OS
        self.backbone_feature_depth = feature_depth
        self.drop_prob = drop_prob
        self.bn_d = bn_d

        # stride play
        self.strides = [2, 2, 2, 2, 2]
        # check current stride
        current_os = 1
        for s in self.strides:
            current_os *= s
        print("Decoder original OS: ", int(current_os))
        # redo strides according to needed stride
        for i, stride in enumerate(self.strides):
            if int(current_os) != self.backbone_OS:
                if stride == 2:
                    current_os /= 2
                    self.strides[i] = 1
                if int(current_os) == self.backbone_OS:
                    break
        print("Decoder new OS: ", int(current_os))
        print("Decoder strides: ", self.strides)

        # decoder
        self.dec5 = self._make_dec_layer(BasicBlock,
                                         [self.backbone_feature_depth, 512],
                                         bn_d = self.bn_d,
                                         stride = self.strides[0])
        self.dec4 = self._make_dec_layer(BasicBlock, [512, 256], bn_d = self.bn_d,
                                         stride = self.strides[1])
        self.dec3 = self._make_dec_layer(BasicBlock, [256, 128], bn_d = self.bn_d,
                                         stride = self.strides[2])
        self.dec2 = self._make_dec_layer(BasicBlock, [128, 64], bn_d = self.bn_d,
                                         stride = self.strides[3])
        # self.dec1 = self._make_dec_layer(BasicBlock, [64, 32], bn_d=self.bn_d,
        #                                  stride=self.strides[4])

        # layer list to execute with skips

        # self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]
        # for a bit of fun
        self.dropout = nn.Dropout2d(self.drop_prob)

        # last channels
        self.last_channels = 64

    def _make_dec_layer(self, block, planes, bn_d = 0.1, stride = 2):
        layers = []

        #  downsample
        if stride == 2:
            layers.append(("upconv", nn.ConvTranspose2d(planes[0], planes[1],
                                                        kernel_size = [1, 4], stride = [1, 2],
                                                        padding = [0, 1])))
        else:
            layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                             kernel_size = 3, padding = 1)))
        layers.append(("bn", nn.BatchNorm2d(planes[1], momentum = bn_d)))
        layers.append(("relu", nn.LeakyReLU(0.1)))

        #  blocks
        layers.append(("residual", block(planes[1], planes, bn_d)))

        return nn.Sequential(OrderedDict(layers))

    def run_layer(self, x, layer, skips, os):
        feats = layer(x)  # up
        if feats.shape[-1] > x.shape[-1]:
            os //= 2  # match skip
            feats = feats + skips[os].detach()  # add skip
        x = feats
        return x, skips, os

    def forward(self, x):

        # run layers
        # x, skips, os = self.run_layer(x, self.dec5, skips, os)
        # x, skips, os = self.run_layer(x, self.dec4, skips, os)
        # x, skips, os = self.run_layer(x, self.dec3, skips, os)
        # x, skips, os = self.run_layer(x, self.dec2, skips, os)
        # x, skips, os = self.run_layer(x, self.dec1, skips, os)
        x = self.dec5(x)
        x = self.dec4(x)
        x = self.dec3(x)
        x = self.dec2(x)
        #x = self.dec1(x)

        x = self.dropout(x)

        return x


class PUP_decoder(nn.Module):

    def __init__(self, num_classes, img_size, patch_size, embed_dim):
        super(PUP_decoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        extra_in_channels = int(self.embed_dim / 4)
        in_channels = [
            self.embed_dim,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
        ]
        out_channels = [
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            self.num_classes,
        ]

        modules = []
        for i, (in_channel, out_channel) in enumerate(
                zip(in_channels, out_channels)
        ):
            modules.append(
                nn.Conv2d(
                    in_channels = in_channel,
                    out_channels = out_channel,
                    kernel_size = 1,
                    stride = 1
                )
            )
            if i != 4:
                modules.append(nn.Upsample(scale_factor = 2, mode = 'bilinear'))
        # self.decode_net = IntermediateSequential(
        #     *modules, return_intermediate = False
        # )
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self._reshape_output(x)
        x = self.decoder(x)
        return x

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            self.embed_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out = True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p = dropout_rate)

        self.dropout2 = nn.Dropout2d(p = dropout_rate)

        self.conv1 = nn.Conv2d(in_filters // 4 + 2 * out_filters, out_filters, (3, 3), padding = 1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation = 2, padding = 2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2, 2), dilation = 2, padding = 1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters * 3, out_filters, kernel_size = (1, 1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p = dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        # upB = torch.cat((upA, skip), dim = 1)
        # if self.drop_out:
        #     upB = self.dropout2(upB)

        upE = self.conv1(upA)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1, upE2, upE3), dim = 1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class Salsadecoder(nn.Module):

    def __init__(self, num_classes, img_size, patch_size, embed_dim):
        super(Salsadecoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out = False)

        self.logits = nn.Conv2d(32, self.num_classes, kernel_size = (1, 1))

    def forward(self, x):
        x = self._reshape_output(x)
        out = self.upBlock1(x)
        out = self.upBlock2(out)
        out = self.upBlock3(out)
        out = self.upBlock4(out)
        return out

    def _reshape_output(self, x):
        B, _, D = x.shape
        x = x.view(
            B,
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            D,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
