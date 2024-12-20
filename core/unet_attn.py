import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules import DoubleConv, Down, Up, SelfAttention

class UNet_Attn(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.class_emb = nn.Linear(num_classes, time_dim)
            self.depth_emb = nn.Linear(num_classes, time_dim)
            self.t_proj = nn.Linear(3 * time_dim, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, class_vector, depth_vector):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        class_emb = self.class_emb(class_vector)
        depth_emb = self.depth_emb(depth_vector)

        t_combined = torch.cat([t, class_emb, depth_emb], dim=-1)
        t_combined = self.t_proj(t_combined)

        x1 = self.inc(x)
        x2 = self.down1(x1, t_combined)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t_combined)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t_combined)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t_combined)
        x = self.sa4(x)
        x = self.up2(x, x2, t_combined)
        x = self.sa5(x)
        x = self.up3(x, x1, t_combined)
        x = self.sa6(x)
        output = self.outc(x)
        return output