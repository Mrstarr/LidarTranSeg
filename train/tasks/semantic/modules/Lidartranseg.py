import torch
import torch.nn as nn
import torch.nn.functional as F
#from .encoder import trunc_normal_, DropPath, Mlp, Attention
#from .decoder import Salsadecoder, LidarDecoder

class Block(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias, qk_scale, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransContext(nn.Module):

    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth,
                 num_heads, qkv_bias=False, qk_scale=None, mlp_drop=0.1, attn_drop=0.,
                 path_drop=0., norm_layer=nn.LayerNorm):
        super(TransContext, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.mlp_drop = mlp_drop
        self.attn_drop = attn_drop
        self.path_drop = path_drop
        self.norm_layer = norm_layer
        self.num_patches = img_size[0] * img_size[1] / patch_size / patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_size**2, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate,
                                                self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer)
            for i in range(self.depth)])
        self.unfold = nn.Unfold(kernel_size = patch_size, stride = patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = 7, padding = 3, stride = 1)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.unfold(x) # B, Cp2, N
        x = x.transpose(1, 2).reshape(B * self.num_patches, C, self.patch_size, self.patch_size) # B*N, C, p, p
        x = self.proj(x) # B*N, dim, p, p
        x = x.reshape(B * self.num_patches, self.embed_dim, -1).transpose(1, 2) # B*N, p*p, dim
        x = x + self.pos_embed
        x = self.pos_drop(x)

        #outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = x.reshape(B, self.num_patches, self.patch_size**2, self.embed_dim).permute(0, 3, 1, 2)
        return x

    def _reshape_output(self, x):
        BN, pp, dim = x.shape
        x = x.reshape(
            x.size(0),
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            self.embed_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class LidarTranSeg(nn.Module):

    def __init__(self,  num_of_classes, in_channels, img_size=(64,2048), patch_size=16, embed_dim=1024, depth=12, head=16):

        super(LidarTranSeg, self).__init__()
        self.encoder = VisionTransformer('vit_base_patch16_384', img_size=img_size, patch_size=patch_size,
                                         in_chans=in_channels, embed_dim=embed_dim, depth=depth, num_heads=head,
                                         num_classes=num_of_classes, drop_rate = 0.1, attn_drop_rate = 0.0,
                                         drop_path_rate = 0.0)
        self.decoder = LidarDecoder(feature_depth=embed_dim)
        self.head = nn.Sequential(nn.Dropout2d(0.01),nn.Conv2d(64, num_of_classes, kernel_size = (3,3), padding = 1))

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        out = self.head(out)
        out = F.softmax(out, dim=1)
        return out


if __name__ == "__main__":
    #encoder_layer = nn.TransformerEncoderLayer(d_model = 512, nhead = 8)
    #transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 6)
    src1 = torch.rand(10, 32, 64, 1024)
    src = F.unfold(src1, kernel_size = 64, stride = 64)
    #print(src.size())
    #out = transformer_encoder(src)
    patch_size = 64
    src =  src.transpose(1, 2).reshape(10 * 16, 32, patch_size, patch_size)
    src = src.reshape(10 * 16, 32, -1).transpose(1, 2)  # B*N, p*p, dim
    src = src.reshape(10, 16, patch_size ** 2, 32).permute(0, 3, 2, 1).reshape(10,-1, 16)
    src = F.fold(src, output_size = (64,1024),kernel_size = (64,64),stride = 64)

    print(torch.equal(src, src1))