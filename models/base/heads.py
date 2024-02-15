import torch.nn as nn

class SegmentationHead(nn.Module):
    """Segmentation head for a segmentation model"""
    def __init__(self, in_channels, out_channels, kernel_size=3, **kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        
    def forward(self, x):
        return self.conv2d(x)
        




