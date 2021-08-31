import torch
import torch.nn as nn


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
