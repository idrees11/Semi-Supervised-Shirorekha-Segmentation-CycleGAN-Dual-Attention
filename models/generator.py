
import torch
import torch.nn as nn
from .dual_attention import DualAttention

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(out_c, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.attention = DualAttention(out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.attention(x)
        p = self.pool(x)
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True)
        )
        self.attention = DualAttention(out_c)
        
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = x + skip
        x = self.attention(x)
        return x

class classifier_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.outputs = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, inputs):
        x = self.outputs(inputs)
        return x
    
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.e1 = encoder_block(in_channels, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        self.outputs = classifier_block(64, out_channels)

    def forward(self, inputs):
        # Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder with skip connections
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        # Output
        x = self.outputs(d4)
        return x
