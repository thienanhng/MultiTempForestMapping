import torch
import torch.nn as nn

from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from typing import Type, Union, List, Optional


class ResNetEncoder(nn.Module):
    """
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock,
        in_channels: int = 3,
        out_channels: List[int] = [64, 64, 128, 256, 512], #first element is the output of the first convolution
        layers: List[int] = [2, 2, 2, 2],
        aux_in_channels: Optional[int] = None,
        aux_in_position: Optional[int] = None,
        init_stride: List[int] = [2, 2],
        bn_momentum=0.1
    ) -> None:
        
        super(ResNetEncoder, self).__init__()

        # check arguments
        if len(out_channels) != len(layers) + 1:
            raise ValueError('len(out_channels) should be len(layers) + 1')
        if len(out_channels) < 2:
            raise ValueError('out_channels should have at least 2 elements so that '
                                'the encoder has at least 1 residual block')

        self.in_channels = in_channels 
        self._out_channels = out_channels
        self.aux_in_position = aux_in_position
        # self._norm_layer = nn.BatchNorm2d
        self.inplanes = out_channels[0] # used by self._make_layer()
        self.dilation = 1
    
        # create layers
        if aux_in_position == 0:
            conv1 = nn.Conv2d(in_channels + aux_in_channels, out_channels[0], kernel_size=7, stride=init_stride[0], padding=3,
                               bias=False)
        else:
            conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=7, stride=init_stride[0], padding=3,
                                bias=False)
        bn1 = nn.BatchNorm2d(out_channels[0], momentum=bn_momentum)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=init_stride[1], padding=1)

        # first block of residual blocks
        if aux_in_position == 1:
            self.inplanes += aux_in_channels
        layer1 = self._make_layer(block, 
                                  out_channels[1], 
                                  layers[0],
                                  bn_momentum=bn_momentum)

        stages = [  nn.Sequential(conv1, bn1, relu),
                    nn.Sequential(maxpool, layer1) ]

        # next blocks of residual blocks
        if len(out_channels) > 2:
            for i in range(2, len(out_channels)):
                if aux_in_position == i:
                    self.inplanes += aux_in_channels
                stages.append(self._make_layer(block, 
                                               out_channels[i], 
                                               layers[i-1], 
                                               stride=2,
                                               bn_momentum=bn_momentum))

        self.stages = nn.ModuleList(stages)

        if aux_in_channels is None:
            self.forward = self._forward_1_input
        else:
            self.forward = self._forward_2_inputs

        # parameter initialization done at SegmentationModel level

    @property
    def out_channels(self): #used by decoder to make u-net skip connections
        """Return channels dimensions for each tensor of forward output of encoder"""
        return [self.in_channels] + self._out_channels

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, bn_momentum:float = 0.1) -> nn.Sequential:
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_momentum),
            )

        layers = []
        new_layer = block(self.inplanes, planes, stride, downsample, 
                            dilation = previous_dilation, norm_layer = norm_layer)
        for _, module in new_layer.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = bn_momentum
        layers.append(new_layer)
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            new_layer = block(self.inplanes, planes, dilation=self.dilation,
                                norm_layer=norm_layer)
            for _, module in new_layer.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.momentum = bn_momentum
            layers.append(new_layer)

        return nn.Sequential(*layers)

    def _forward_1_input(self, x):
        """
        Forward method for a single input
        """
        features = []
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            features.append(x) 
        return features

    def _forward_2_inputs(self, x_0, x_1):
        """
        Forward method for 2 inputs (main input + auxiliary input)
        Not the fastest (if statement for each stage) but flexible
        """
        features = []
        x = x_0
        for i in range(len(self.stages)):
            if self.aux_in_position == i:
                x = torch.cat((x, x_1), dim = 1)
            x = self.stages[i](x)
            features.append(x)
        return features


