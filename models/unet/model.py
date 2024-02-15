from typing import List
from .decoder import UnetDecoder
from ..encoders import ResNetEncoder
from ..base import SegmentationModel
from ..base import SegmentationHead
from ..gru import GRU, IrregGRU
import torch
from torch.nn import Module
import torch.nn as nn

class Unet(SegmentationModel):
    """
    Unet model with a ResNet-18-like encoder, and possibly an auxiliary input source
    """

    def __init__(
        self,
        encoder_depth: int = 4,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        out_channels: int = 1,
        upsample: bool = False,
        aux_in_channels: int = 0,
        aux_in_position: int = 1,
        init_stride = [2, 2],
        bn_momentum=0.1,
        **kwargs
    ):
        """
        Args:
            - encoder_depth: number of blocks in the ResNet encoder, each block itself 
                containing 2 correction blocks (Resnet-18). encoder_depth does not include
                    the initial conv and maxpool layers
            - decoder_channels: number of output channels of each decoder layer
            - in_channels: number of channels of the main input
            - out_channels: number of output channels
            - upsample: whether to upsample or not the activations at the end of each 
                decoder block. The upsampling is done via a transposed convolution with 
                upsampling factor 2. If a single value is specified the same value will
                be used for all decoder blocks.
            - aux_in_channels: number of channels of the auxiliary input
            - aux_in_position: position of the auxiliary input in the model:
                0: concatenated to the main input before entering the model
                1: before the 1st block of the encoder
                2: before the 2nd block of the encoder
                3: etc.
            - init_stride (list of ints): stride value of the first convolutional layer and the first maxpool layer of 
                the encoder. The output has a resolution (init_stride[0]*init_stride[1]) times lower than the input.
            - bn_momentum: momentum of the batch normalization layers
        """
        
        super().__init__()
        self.last_decoder_channels = decoder_channels[-1] # used by GRUUnet
        layers, decoder_out_channels = self.set_channels(aux_in_channels, aux_in_position, encoder_depth)
        encoder, decoder, segmentation_head = self._get_model_blocks()
        self.encoder = encoder(in_channels=in_channels,
                        aux_in_channels=aux_in_channels,
                        out_channels=decoder_out_channels,
                        layers=layers,
                        aux_in_position=aux_in_position,
                        init_stride=init_stride,
                        bn_momentum=bn_momentum)

        self.decoder = decoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            upsample=upsample,
            use_bn=True,
            bn_momentum=bn_momentum
        )
        self.segmentation_head = segmentation_head(
            in_channels=decoder_channels[-1], 
            out_channels=out_channels,
            kernel_size=3,
            **kwargs
        )
        
        if out_channels == 1:
            self.seg_normalization = nn.Sigmoid()
        else:
            self.seg_normalization = nn.Softmax(dim = 1)
            

        self.initialize()

    def _get_model_blocks(self):
        return ResNetEncoder, UnetDecoder, SegmentationHead

    def set_channels(self, aux_in_channels, aux_in_position, encoder_depth):
        if (aux_in_channels is None) != (aux_in_position is None):
            raise ValueError('aux_in_channels and aux_in_position should be both specified')
        # architecture based on Resnet-18
        out_channels = [64, 64, 128, 256, 512]
        out_channels = out_channels[:encoder_depth+1]
        if (aux_in_position is not None) and (aux_in_position > 0):
            out_channels[aux_in_position] += aux_in_channels
        layers = [2] * (len(out_channels)-1)
        return layers, out_channels
    
class NonRecurrentUnet(Module):
    """ Takes a time series as input and feeds each time stamp to a Unet model independently """
    def __init__(   
        self,
        encoder_depth: int=4,
        decoder_channels: List[int]=(256, 128, 64, 32, 16),
        in_channels: int=3,
        out_channels: int=1,
        upsample: bool=False,
        aux_in_channels: int=0,
        aux_in_position: int=1,
        in_out_scale: List[int]=4,
        bn_momentum=0.1,
        **kwargs):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.unet_in_channels = self.in_channels
        
        self.unet = Unet(encoder_depth,
                        decoder_channels,
                        self.unet_in_channels,
                        self.out_channels,
                        upsample,
                        aux_in_channels,
                        aux_in_position,
                        bn_momentum=bn_momentum,
                        **kwargs) 
        
        self.in_out_scale = in_out_scale
        
        if in_out_scale != 1:
            self.upsample = torch.nn.Upsample(scale_factor=in_out_scale, mode='nearest') 
        else:
            self.upsample = torch.nn.Identity()
            
        self.seg_normalization = self.unet.seg_normalization
        
    def forward(self, temp_data, static_data, temporal_mask=None, *args, **kwargs):
        """
        temp_data:      list of n_t tensors of  (batch_size, n_channels, h,     w)
        static_data:                            (batch_size, n_channels, h,     w)
        output:                                 (batch_size, n_classes,  n_t,   h, w)
        temporal_mask: boolean 1-D tensor indicating for each time step if the inputs are valid
        """
        batch_size = static_data.shape[0]
        h = max(max([data.shape[-2] for data in temp_data]), static_data.shape[-2])
        w = max(max([data.shape[-1] for data in temp_data]), static_data.shape[-1])
        h_out, w_out = int(h//self.in_out_scale), int(w//self.in_out_scale)
        n_steps = len(temp_data)
        if temporal_mask == None: 
            temporal_mask = torch.full((n_steps,), True, dtype=torch.bool) 
        # create tensor to store outputs
        output = static_data.new_zeros(batch_size, self.out_channels, n_steps, h_out, w_out)
        # first forward pass
        valid_steps = torch.nonzero(temporal_mask)
        i_start = torch.min(valid_steps).item() # first valid step
        it = range(i_start, n_steps)
        for i in it:
            if temporal_mask[i]: 
                temp_input = temp_data[i]
                output[:, :, i] = self.unet(temp_input, static_data)[0] 
        return output
    
class RecurrentUnet(Module):
    """ 
    Takes a times series as input and feeds each input as well as the previous output to a Unet model
    """
    def __init__(self,
        encoder_depth: int=4,
        decoder_channels: List[int]=(256, 128, 64, 32, 16),
        in_channels: int=3,
        out_channels: int=1,
        upsample: bool=False,
        aux_in_channels: int=0,
        aux_in_position: int=1,
        in_out_scale: List[int]=4,
        bn_momentum=0.1,
        reverse=False,
        rec_init='zero', #'copy_input' or 'zero'
        temp_loop=True,
        rec_features_norm='batchnorm',
        rec_features_clamp='sigmoid',
        rec_features_clamp_val=3,
        **kwargs):
        
        super().__init__()
        self.temp_loop = temp_loop
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if self.temp_loop:
            self.unet_in_channels = self.in_channels + self.out_channels
        else:
            self.unet_in_channels = self.in_channels
        
        self.unet = Unet(encoder_depth,
                        decoder_channels,
                        self.unet_in_channels,
                        self.out_channels,
                        upsample,
                        aux_in_channels,
                        aux_in_position,
                        bn_momentum=bn_momentum,
                        **kwargs) 
        
        self.in_out_scale = in_out_scale
        self.reverse = reverse
        self.rec_init = rec_init
        
        if temp_loop:
            if rec_features_norm == 'none':
                self.rec_features_norm = torch.nn.Identity()
            elif rec_features_norm == 'batchnorm':
                self.rec_features_norm = torch.nn.BatchNorm2d(num_features=out_channels)
            else:
                raise NotImplementedError(
                    'Recursion features rescaling with {} not implemented'.format(rec_features_norm)
                                        )
            
            if rec_features_clamp == 'clamp':
                self.clamp = lambda x: torch.clamp(x, min=-rec_features_clamp_val, max=rec_features_clamp_val)
            elif rec_features_clamp == 'sigmoid':
                self.clamp = lambda x: 2 * torch.sigmoid(x) - 1
            else:
                raise NotImplementedError(
                    'Recursion features clamping with {} not implemented'.format(rec_features_clamp)
                                        )
        
        if in_out_scale != 1:
            self.upsample = torch.nn.Upsample(scale_factor=in_out_scale, mode='nearest') 
        else:
            self.upsample = torch.nn.Identity()
        
    def forward(self, temp_data, static_data, temporal_mask=None, *args, **kwargs):
        """
        temp_data:      list of n_t tensors of  (batch_size, n_channels, h,     w)
        static_data:                            (batch_size, n_channels, h,     w)
        output:                                 (batch_size, n_classes,  n_t,   h, w)
        temporal_mask: boolean 1-D tensor indicating for each time step if the inputs are valid
        """
        batch_size = static_data.shape[0]
        h = max(max([data.shape[-2] for data in temp_data]), static_data.shape[-2])
        w = max(max([data.shape[-1] for data in temp_data]), static_data.shape[-1])
        h_out, w_out = int(h//self.in_out_scale), int(w//self.in_out_scale)
        n_steps = len(temp_data)
        if temporal_mask == None: 
            temporal_mask = torch.full((n_steps,), True, dtype=torch.bool)
        # create tensor to store outputs
        output = static_data.new_zeros(batch_size, self.out_channels, n_steps, h_out, w_out)
        # first forward pass
        valid_steps = torch.nonzero(temporal_mask)
        if self.reverse:
            i_start = torch.max(valid_steps).item() # last valid step
        else:
            i_start = torch.min(valid_steps).item() # first valid step
        
        if self.temp_loop:
            if self.rec_init == 'copy_input':
                # does not work for multiband inputs
                dummy_output = (temp_data[i_start]*(-1)).repeat(1, self.out_channels, 1, 1) 
            else:
                dummy_output = static_data.new_zeros((batch_size, self.out_channels, h, w))
            temp_input = torch.cat((temp_data[i_start], dummy_output), dim=1)
        else:
            temp_input = temp_data[i_start]
        output[:, :, i_start] = self.unet(temp_input, static_data) 
        last_i = i_start
        if self.reverse:
            it = range(i_start-1, -1, -1)
        else:
            it = range(i_start + 1, n_steps)
        for i in it:
            if temporal_mask[i]: 
                if self.temp_loop:
                    rec = self.upsample(self.rec_features_norm(self.clamp(torch.clone(output[:, :, last_i]))))
                    temp_input = torch.cat((temp_data[i], rec), dim=1)
                else:
                    temp_input = temp_data[i]
                output[:, :, i] = self.unet(temp_input, static_data) 
                last_i = i
        return output
     
class GRUUnet(Module):
    """ Applies a Unet to each time stamp, and feeds the obtained Unet output/features to a ConvGRU """
    def __init__(self,
            encoder_depth: int=4,
            decoder_channels: List[int]=(256, 128, 64, 32, 16),
            in_channels: int=3,
            unet_out_channels: int=1,
            out_channels: int=1,
            upsample: bool=False,
            aux_in_channels: int=0,
            aux_in_position: int=1,
            in_out_scale: List[int]=4,
            bn_momentum=0.1,
            gru_irreg=False,
            gru_input: str='logits',
            gru_reset_channels: int=1,
            gru_update_channels: int=1,
            gru_kernel_size: int=3,
            gru_init:str='last',
            gru_norm_dt:bool=True,
            reverse=False,
            **kwargs):
        
        super().__init__() 
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet_out_channels = unet_out_channels
        self.in_out_scale = in_out_scale
        self.reverse = reverse
        
        self.unet = Unet(encoder_depth,
                        decoder_channels,
                        self.in_channels,
                        self.unet_out_channels,
                        upsample,
                        aux_in_channels,
                        aux_in_position,
                        bn_momentum=bn_momentum,
                        **kwargs)
        
        gru_last_actv = self.unet.seg_normalization
        
        Wh_template = None
        if gru_input == 'logits':
            gru_in_channels = unet_out_channels
            self.get_gru_input = lambda pred, *features: pred
        elif gru_input == 'df':
            gru_in_channels = self.unet.last_decoder_channels
            self.get_gru_input = lambda pred, decoder_features, *other_features: decoder_features
            Wh_template = self.unet.segmentation_head.conv2d
        else:
            raise NotImplementedError    
        
        self.gru_irreg = gru_irreg
        if self.gru_irreg:
            self.gru = IrregGRU(in_channels=gru_in_channels,
                        reset_channels=gru_reset_channels,
                        update_channels=gru_update_channels,
                        out_channels=out_channels,
                        kernel_size=gru_kernel_size,
                        last_actv=gru_last_actv,
                        bias=True,
                        Wh_template=Wh_template,
                        norm_dt=gru_norm_dt)
        else:
            self.gru = GRU(in_channels=gru_in_channels,
                        reset_channels=gru_reset_channels,
                        update_channels=gru_update_channels,
                        out_channels=out_channels,
                        kernel_size=gru_kernel_size,
                        last_actv=gru_last_actv,
                        bias=True,
                        Wh_template=Wh_template)
        
        self.gru.initialize(init_mode=gru_init, 
                            input_mode=gru_input)
        
        # normalization function of the Unet is already applied inside the GRU
        # only rescaling might be necessary
        if out_channels == 1: # 2 classes
            self.seg_normalization = lambda x: x
        else:
            raise NotImplementedError
    
    def forward(self, 
                temp_data, 
                static_data, 
                years=None, 
                temporal_mask=None, 
                opt_outputs = False, 
                *args, 
                **kwargs):
        batch_size = static_data.shape[0]
        h = max(max([data.shape[-2] for data in temp_data]), static_data.shape[-2])
        w = max(max([data.shape[-1] for data in temp_data]), static_data.shape[-1])
        h_out, w_out = int(h//self.in_out_scale), int(w//self.in_out_scale)
        n_steps = len(temp_data)
        
        # time steps to be ignored
        if temporal_mask == None: 
            temporal_mask = torch.full((n_steps,), True, dtype=torch.bool) 
            
        # determine processing order
        valid_steps = torch.nonzero(temporal_mask)
        if self.reverse:
            i_start = torch.max(valid_steps).item() # last valid step
            it = range(n_steps-1, -1, -1) 
        else:
            i_start = torch.min(valid_steps).item() # first valid step
            it = range(0, n_steps) 

        # create list to store outputs
        output = torch.zeros((batch_size, self.out_channels, n_steps, h_out, w_out), 
                             dtype=torch.float32, 
                             device=static_data.device)
        if opt_outputs:
            x = torch.zeros_like(output)
            r = torch.zeros_like(output)
            h_hat = torch.zeros_like(output)
            z = torch.zeros_like(output)
        new_z = None
        for i in it:
            if temporal_mask[i]: 
                temp_input = temp_data[i]
                # feature extraction
                unet_pred, decoder_features, encoder_features = self.unet(temp_input, static_data)
                if opt_outputs:
                    x[:, :, i] = self.unet.seg_normalization(unet_pred) 
                # pass features to gru
                new_gru_input = self.get_gru_input(unet_pred, decoder_features, encoder_features, temp_input, static_data)
                if i == i_start:
                    new_output = self.gru(new_gru_input, None) 
                    last_years = years[:, i]
                else:
                    current_years = years[:, i]
                    t_interv = torch.abs(current_years - last_years)
                    new_output, new_z, new_h_hat, new_r = self.gru(new_gru_input, new_output, t_interv) 
                    if opt_outputs:
                        r[:, :, i] = new_r
                        h_hat[:, :, i] = new_h_hat
                        z[:, :, i] = new_z
                    last_years = current_years
                       
                output[:, :, i] = new_output
                 
        if opt_outputs:  
            return output, x, z, h_hat, r
        return output 