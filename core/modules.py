import torch
import torch.nn as nn
import torch.nn.functional as F

# class SelfAttention(nn.Module):
#     def __init__(self, channels, size):
#         super(SelfAttention, self).__init__()
#         self.channels = channels
#         self.size = size
#         self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
#         self.ln = nn.LayerNorm([channels])
#         self.ff_self = nn.Sequential(
#             nn.LayerNorm([channels]),
#             nn.Linear(channels, channels),
#             nn.GELU(),
#             nn.Linear(channels, channels),
#         )

#     def forward(self, x):
#         x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
#         x_ln = self.ln(x)
#         attention_value, _ = self.mha(x_ln, x_ln, x_ln)
#         attention_value = attention_value + x
#         attention_value = self.ff_self(attention_value) + attention_value
#         return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

class SelfAttention(nn.Module):
    def __init__(self, channels, size, num_heads=4):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert (
            self.channels % num_heads == 0
        ), "Channels must be divisible by the number of heads."

        # Learnable weights for query, key, and value
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        # Final output projection
        self.out_proj = nn.Linear(channels, channels)

        # Layer norm and feed-forward network
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # Reshape input from (B, C, H, W) to (B, N, C), where N = H * W
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # Shape: (B, N, C)

        # Apply layer norm
        x_ln = self.ln(x)

        # Compute Q, K, V
        Q = self.q_proj(x_ln)  # Shape: (B, N, C)
        K = self.k_proj(x_ln)  # Shape: (B, N, C)
        V = self.v_proj(x_ln)  # Shape: (B, N, C)

        # Reshape for multi-head attention: (B, N, C) -> (B, num_heads, N, head_dim)
        Q = Q.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Compute scaled dot-product attention
        scale = self.head_dim ** -0.5
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # Shape: (B, num_heads, N, N)
        attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: (B, num_heads, N, N)
        attention_output = torch.matmul(attention_weights, V)  # Shape: (B, num_heads, N, head_dim)

        # Concatenate heads and project back to original dimensions
        attention_output = (
            attention_output.permute(0, 2, 1, 3).contiguous().view(B, -1, self.channels)
        )  # Shape: (B, N, C)
        attention_output = self.out_proj(attention_output)  # Shape: (B, N, C)

        # Add residual connection
        attention_output = attention_output + x

        # Apply feed-forward network and add another residual connection
        attention_output = self.ff_self(attention_output) + attention_output

        # Reshape back to (B, C, H, W)
        return attention_output.permute(0, 2, 1).view(B, C, H, W)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False, dropout_prob=0.0):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        ]

        if dropout_prob > 0:
            layers.insert(3, nn.Dropout(dropout_prob))  # Add dropout after activation

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, dropout_prob=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True, dropout_prob=dropout_prob),
            DoubleConv(in_channels, out_channels, dropout_prob=dropout_prob),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, dropout_prob=0.0):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True, dropout_prob=dropout_prob),
            DoubleConv(in_channels, out_channels, in_channels // 2, dropout_prob=dropout_prob),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
