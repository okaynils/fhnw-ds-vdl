import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules import DoubleConv, Down, Up, SelfAttention

class UNet_Baseline(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.bot = DoubleConv(128, 128)
        
        self.up1 = Up(192, 64)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.class_emb = nn.Linear(num_classes, time_dim)
            self.depth_emb = nn.Linear(num_classes, time_dim)
            self.t_proj = nn.Linear(3 * time_dim, time_dim)
        else:
            self.class_emb = nn.Identity()
            self.depth_emb = nn.Identity()
            self.t_proj = nn.Identity()

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
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
        x_bot = self.bot(x2)

        x = self.up1(x_bot, x1, t_combined)
        output = self.outc(x)

        return output
