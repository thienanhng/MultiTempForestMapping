import torch
import torch.nn as nn

from ..base import modules as md


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            upsample = True,
            use_bn=True,
            bn_momentum=0.1
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            use_bn=use_bn,
            bn_momentum=bn_momentum
        )
        
        if upsample:
            conv2_out_channels = in_channels
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
            
        else:
            conv2_out_channels = out_channels
            self.upsample = nn.Identity()

        self.conv2 = md.Conv2dReLU(
            in_channels,
            conv2_out_channels,
            kernel_size=3,
            padding=1,
            use_bn=use_bn,
            bn_momentum=bn_momentum
        )

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 use_bn=True,
                 bn_momentum=0.1):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_bn=use_bn,
            bn_momentum=bn_momentum
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_bn=use_bn,
            bn_momentum=bn_momentum
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            upsample = True,
            use_bn=True,
            bn_momentum=0.1,
            # center=False,
    ):
        super().__init__()

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = [0] + list(encoder_channels[1:]) 
        out_channels = decoder_channels

        # duplicate the upsample option if a single boolean has been provided
        if isinstance(upsample, bool):
            upsample = [upsample] * len(in_channels)

        self.blocks = nn.ModuleList(self._get_blocks(in_channels, 
                                                     skip_channels, 
                                                     out_channels, 
                                                     upsample, 
                                                     use_bn, 
                                                     bn_momentum))

    def _get_blocks(self, in_channels, skip_channels, out_channels, upsample, use_bn, bn_momentum):
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, upsamp, use_bn=use_bn, bn_momentum=bn_momentum)
            for in_ch, skip_ch, out_ch, upsamp in zip(in_channels, skip_channels, out_channels, upsample)
        ]
        return blocks

    def forward(self, *features):

        features = features[::-1]  # reverse channels to start from head of encoder

        # head = features[0]
        x = features[0]
        skips = (None,) + features[1:]

        # x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

