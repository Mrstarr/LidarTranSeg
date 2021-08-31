import torch
import torch.nn as nn
import torch.nn.functional as F
from train.tasks.semantic.modules.encoder import VisionTransformer
from train.tasks.semantic.modules.decoder import PUP_decoder

class LidarTranSeg(nn.Module):

    def __init__(self,  num_of_classes, in_channels, img_size=(64,2048), patch_size=16, embed_dim=784, depth=12):

        super(LidarTranSeg, self).__init__()
        self.encoder = VisionTransformer('vit_base_patch16_384', img_size=img_size, patch_size=patch_size,
                                         in_chans=in_channels, embed_dim=embed_dim, depth=depth, num_heads=16,
                                         num_classes=num_of_classes, drop_rate = 0.1, attn_drop_rate = 0.0,
                                         drop_path_rate = 0.0)
        self.decoder = PUP_decoder(num_classes=num_of_classes, img_size=img_size, patch_size=patch_size,
                                   embed_dim=embed_dim)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        out = F.softmax(out, dim=1)
        return out
