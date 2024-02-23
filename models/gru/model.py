import torch
import torch.nn as nn
from torch.nn import Module
from copy import deepcopy

class GRU(Module):
    """
    Convolutional Gated Recurrent Unit
    """
    def __init__(self, 
                 in_channels: int=1,
                 reset_channels: int=1,
                 update_channels:int=1,
                 out_channels:int=1,
                 bias:bool=True,
                 kernel_size:int=3,
                 last_actv:nn.Module=torch.sigmoid,
                 Wh_template:nn.Module=None):
        super().__init__()
        
        padding = kernel_size//2
        self.last_actv = last_actv
        self.bias = bias
        
        #input to update
        self.Wz = nn.Conv2d(in_channels=in_channels,
                            out_channels=update_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=bias) 
            
        # output to update
        self.Uz = nn.Conv2d(in_channels=out_channels,
                            out_channels=update_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=bias)
        
        #input to reset
        self.Wr = nn.Conv2d(in_channels=in_channels,
                            out_channels=reset_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=bias) 
        # output to reset
        self.Ur = nn.Conv2d(in_channels=out_channels,
                            out_channels=reset_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=bias)   
        
        #input to candidate
        if Wh_template is None:
            self.Wh = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=bias) 
        else:
            self.Wh = deepcopy(Wh_template)
        # output to candidate
        self.Uh = nn.Conv2d(in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=bias) 
            
    def initialize(self, init_mode='last', input_mode='df'):  
        std = 1e-3    
        nn.init.normal_(self.Wz.weight,  mean=0, std=std) # to initialize z as 1s
        nn.init.normal_(self.Uz.weight,  mean=0, std=std) # to initialize z as 1s
        if input_mode == 'logits':
            # initialize h_hat as x
            nn.init.dirac_(self.Wh.weight) 
            self.Wh.weight = nn.parameter.Parameter(data = self.Wh.weight + torch.randn_like(self.Wh.weight)*std)
        elif input_mode == 'df':
            pass
        else:
            raise NotImplementedError
        # initialize h_hat as x
        nn.init.normal_(self.Uh.weight,  mean=0, std=std) 
        if self.bias:
            if init_mode == 'last':
                # to initialize z as 0.9 (previous time step is almost ignored)
                nn.init.normal_(self.Wz.bias,  mean=2, std=std) 
                nn.init.normal_(self.Uz.bias,  mean=2, std=std) 
            elif init_mode == 'average':
                # initialize z as 0.9 (previous time step is almost ignored)
                nn.init.normal_(self.Wz.bias,  mean=0, std=std) 
                nn.init.normal_(self.Uz.bias,  mean=0, std=std)
            
            if input_mode == 'logits':
                # initialize h_hat as x
                nn.init.normal_(self.Wh.bias,  mean=0, std=std) 
            elif input_mode == 'df':
                pass
            else:
                raise NotImplementedError
            # initialize h_hat as x
            nn.init.normal_(self.Uh.bias,  mean=0, std=std) 
        
        
    def forward(self, input, last_output=None, t_interv=None, *args, **kwargs):
        if last_output is None:
            h = self.last_actv(self.Wh(input)) 
            return h
        else:
            non_valid_t_interv = torch.isnan(t_interv)
            # update
            z = torch.sigmoid(self.Wz(input) + self.Uz(last_output))  
            # reset
            r = torch.sigmoid(self.Wr(input) + self.Ur(last_output))
            # candidate
            h_hat = self.last_actv(self.Wh(input) + self.Uh(r * last_output))
            # output
            h = z * h_hat + (1-z) * last_output 
            if torch.any(non_valid_t_interv):
                # process some patches as first step of time series
                h[non_valid_t_interv] = self.last_actv(self.Wh(input[non_valid_t_interv]))             
            return h, z, h_hat, r
        
class IrregGRU(GRU):
    """Convolutional Gated Recurrent Unit with time intervals as additional inputs"""
    def __init__(self, 
                 in_channels: int=1,
                 reset_channels: int=1,
                 update_channels: int=1,
                 out_channels: int=1,
                 bias: bool=True,
                 kernel_size: int=3,
                 last_actv: nn.Module=torch.sigmoid,
                 Wh_template: nn.Module=None,
                 norm_dt: bool=True,
                 norm_val: float=5.82):
        """
        - norm_dt: whether to scale the update and reset gates by the time intervals
        - norm_val: value to normalize the time intervals by
        """
        super().__init__(in_channels=in_channels,
                         reset_channels=reset_channels,
                         update_channels=update_channels,
                         out_channels=out_channels,
                         bias=bias,
                         kernel_size=kernel_size,
                         last_actv=last_actv,
                         Wh_template=Wh_template)
        
        self.out_channels = out_channels
        self.eps = 1e-10
        if norm_dt:
            self.norm_val = norm_val
        else:
            self.norm_val = 1.0
   
    def forward(self, input, last_output=None, t_interv=None, *args, **kwargs):
        if last_output is None:
            # first step of time series
            h = self.last_actv(self.Wh(input)) 
            return h
        else:
            non_valid_t_interv = torch.isnan(t_interv)
            t_interv = t_interv.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 
                                                                                 self.out_channels, 
                                                                                 *input.shape[-2:]) + self.eps
            # update
            z = torch.sigmoid((self.Wz(input) + self.Uz(last_output)) * t_interv / self.norm_val) 
            # reset
            r = torch.sigmoid((self.Wr(input) + self.Ur(last_output)) / t_interv * self.norm_val) 
            # candidate
            h_hat = self.last_actv(self.Wh(input) + self.Uh(r * last_output))
            # output
            h = z * h_hat + (1-z) * last_output 
            if torch.any(non_valid_t_interv):
                # process some patches as first step of time series
                h[non_valid_t_interv] = self.last_actv(self.Wh(input[non_valid_t_interv])) 
            return h, z, h_hat, r