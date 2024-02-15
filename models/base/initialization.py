import torch.nn as nn

def initialize_decoder(module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            # Pytorch Resnet implementation uses "fan_out"
            # (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
            # segmentation_models_pytorch uses "fan_in"
            # (https://github.com/qubvel/segmentation_models.pytorch)
            nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

