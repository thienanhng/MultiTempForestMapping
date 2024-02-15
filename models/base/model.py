import torch
from . import initialization as init

class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, *x):
        """
        Pass x through model's encoder, decoder and heads
        """
        encoder_features = self.encoder(*x)
        decoder_output = self.decoder(*encoder_features)
        output = self.segmentation_head(decoder_output)
        return output, decoder_output, encoder_features

