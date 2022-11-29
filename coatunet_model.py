import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torchvision import transforms


def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)

class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class Convblock(nn.Module):
    
      def __init__(self,input_channel,output_channel,kernal=1,stride=1):
            
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernal,stride),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel,output_channel,kernal),
            nn.ReLU(inplace=True),
        )
    

      def forward(self,x):
        x = self.convblock(x)
        return x

class CoAtUNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=29, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.s = Convblock(in_channels,channels[0])

        self.s0 = self._make_layer(
            conv_3x3_bn, channels[0], channels[1], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[1], channels[2], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[2], channels[3], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[3], channels[4], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[4], channels[5], num_blocks[4], (ih // 32, iw // 32))
        
        self.neck = nn.Conv2d(channels[5],channels[6],1,1)
        self.upconv5 = nn.ConvTranspose2d(channels[6],channels[5],3,2,0,1)
        self.dconv5 = Convblock(2*channels[5],channels[5])
        self.upconv4 = nn.ConvTranspose2d(channels[5],channels[4],3,2,0,1)
        self.dconv4 = Convblock(2*channels[4],channels[4])
        self.upconv3 = nn.ConvTranspose2d(channels[4],channels[3],3,2,0,1)
        self.dconv3 = Convblock(2*channels[3],channels[3])
        self.upconv2 = nn.ConvTranspose2d(channels[3],channels[2],3,2,0,1)
        self.dconv2 = Convblock(2*channels[2],channels[2])
        self.upconv1 = nn.ConvTranspose2d(channels[2],channels[1],3,2,0,1)
        self.dconv1 = Convblock(2*channels[1],channels[1])
        self.upconv0 = nn.ConvTranspose2d(channels[1],channels[0],3,2,0,1)
        self.dconv0 = Convblock(2*channels[0],channels[0])
        self.out = nn.Conv2d(channels[0],num_classes,1,1)

    def crop(self,input_tensor,target_tensor):
        # For making the size of the encoder conv layer and the decoder Conv layer same
        _,_,H,W = target_tensor.shape
        return transforms.CenterCrop([H,W])(input_tensor)
        
    def forward(self, x):
        # print('x: ', x.shape)
        x0 = self.s(x)
        # print('x0: ', x0.shape)
        x1 = self.s0(x0)
        # print('x1: ', x1.shape)
        x2 = self.s1(x1)
        # print('x2: ', x2.shape)
        x3 = self.s2(x2)
        # print('x3: ', x3.shape)
        x4 = self.s3(x3)
        # print('x4: ', x4.shape)
        x5 = self.s4(x4)
        # print('x5: ', x5.shape)

        neck = self.neck(x5)
        # print('neck: ', neck.shape)
        
        upconv5 = self.upconv5(neck)
        # print('upconv5: ', upconv5.shape)
        croped = self.crop(upconv5, x5)
        # print('croped: ', croped.shape)
        dconv5 = self.dconv5(torch.cat([croped,x5],1))
        # print('dconv5: ', dconv5.shape)
        upconv4 = self.upconv4(dconv5)
        # print('upconv4: ', upconv4.shape)
        croped = self.crop(upconv4, x4)
        # print('croped: ', croped.shape)
        dconv4 = self.dconv4(torch.cat([croped,x4],1))
        # print('dconv4: ', dconv4.shape)
        upconv3 = self.upconv3(dconv4)
        # print('upconv3: ', upconv3.shape)
        croped = self.crop(upconv3, x3)
        # print('croped: ', croped.shape)
        dconv3 = self.dconv3(torch.cat([croped,x3],1))
        # print('dconv3: ', dconv3.shape)
        upconv2 = self.upconv2(dconv3)
        # print('upconv2: ', upconv2.shape)
        croped = self.crop(upconv2,x2)
        # print('croped: ', croped.shape)
        dconv2 = self.dconv2(torch.cat([croped,x2],1))
        # print('dconv2: ', dconv2.shape)
        upconv1 = self.upconv1(dconv2)
        # print('upconv1: ', upconv1.shape)
        croped = self.crop(upconv1,x1)
        # print('croped: ', croped.shape)
        dconv1 = self.dconv1(torch.cat([croped,x1],1))
        # print('dconv1: ', dconv1.shape)
        upconv0 = self.upconv0(dconv1)
        # print('upconv0: ', upconv0.shape)
        croped = self.crop(upconv0, x0)
        # print('croped: ', croped.shape)
        dconv0 = self.dconv0(torch.cat([croped,x0],1))
        # print('dconv0: ', dconv0.shape)
        out = self.out(dconv0)
        # print('out: ', out.shape)

        return out
        

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)