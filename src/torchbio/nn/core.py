import torch
import torch.nn as nn
import torch.nn.functional as F

import torchbio

class StrandSpecificConv1d(nn.Module):

    def __init__(self, strand: str, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        """strand: foward / reverse
        """
        super().__init__()
        self.strand = strand
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    
    def forward(self, x: torchbio.SeqTensor):
        if self.strand == 'forward':
            return self.conv(x)
        else:
            return self.conv(x.revcomp())

class RevCompConv1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.w = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.ones(out_channels))
        self.padding = padding

    def _reverse_complement_kernel(self, kernel):
        """Reverse complement 1D kernel of shape (out_channels, in_channels, kernel_size)
        """
        return torch.flip(kernel, dims=[1, 2])

    def forward(self, x: torchbio.SeqTensor):
        """x : (bsz, 4, L)
        """
        w_rc = self._reverse_complement_kernel(self.w)
        new_w = torch.cat([self.w, w_rc], axis=0)

        new_bias = torch.cat([self.bias, self.bias], axis=0)

        out = F.conv1d(x, new_w, new_bias, padding=self.padding)
        out_x, out_x_rc = out[:, :self.out_channels, :], out[:, self.out_channels:, :]
        return out_x, out_x_rc