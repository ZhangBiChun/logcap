import torch
import torch.nn as nn
import torch.nn.functional as F

class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_state + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        x_and_res = x_and_res.transpose(1, 2)
        
        x = x_and_res[:, :self.d_inner, :]
        res = x_and_res[:, self.d_inner:, :]

        x = self.conv1d(x)[:, :, :l]
        x = F.silu(x)
        
        x = x.transpose(1, 2)
        x_proj = self.x_proj(x)
        
        delta, B, C = torch.split(x_proj, [self.d_inner, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))
        
        # simplified recurrence for Pure PyTorch speed
        y = x * delta
        
        y = y * F.silu(res.transpose(1, 2))
        return self.out_proj(y)
