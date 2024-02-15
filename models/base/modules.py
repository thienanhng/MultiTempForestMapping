import torch.nn as nn

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_bn=True,
            bn_momentum=0.1
            
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_bn),
        )
        relu = nn.ReLU(inplace=True)

        if use_bn:
            bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, relu, bn)#(conv, bn, relu)
