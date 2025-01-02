import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules import DoubleConv, Down, Up

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda", dropout_prob=0.0):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.dropout_prob = dropout_prob

        self.inc = DoubleConv(c_in, 64, dropout_prob=dropout_prob)
        self.down1 = Down(64, 128, dropout_prob=dropout_prob)
        self.down2 = Down(128, 256, dropout_prob=dropout_prob)
        self.down3 = Down(256, 256, dropout_prob=dropout_prob)

        self.bot1 = DoubleConv(256, 512, dropout_prob=dropout_prob)
        self.bot2 = DoubleConv(512, 512, dropout_prob=dropout_prob)
        self.bot3 = DoubleConv(512, 256, dropout_prob=dropout_prob)

        self.up1 = Up(512, 128, dropout_prob=dropout_prob)
        self.up2 = Up(256, 64, dropout_prob=dropout_prob)
        self.up3 = Up(128, 64, dropout_prob=dropout_prob)
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
        x3 = self.down2(x2, t_combined)
        x4 = self.down3(x3, t_combined)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t_combined)
        x = self.up2(x, x2, t_combined)
        x = self.up3(x, x1, t_combined)
        output = self.outc(x)
        return output