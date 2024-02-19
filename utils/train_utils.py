import numpy as np
from numpy.core.numeric import NaN
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.exp_utils import I_NODATA_VAL

def fit(model, 
        device, 
        dataloader, 
        optimizer, 
        n_batches, 
        seg_criterion, 
        *args, 
        **kwargs):
    """
    Runs 1 training epoch, with auxiliary targets.
    n_batches is used to setup the progress bar only. The actual number of batches is defined by the dataloader.
    Returns average losses over the batches. Warning: the average assumes the same number of pixels contributed to the
    loss for each batch (unweighted average over batches)
    """
    model.train()
    total_losses = []
    seg_losses = []
    running_loss = 0.0
    dump_period = 25 # period (in number of batches) for printing running loss

    # train batch by batch
    
    progress_bar = tqdm(enumerate(dataloader), total=n_batches)
    for batch_idx, data in progress_bar:    
        total_loss = torch.tensor([0.], requires_grad=True, device=device)
        inputs, target = data
                
        inputs = [d.to(device) for d in inputs]
        target = target.to(device)
        # collect outputs of forward pass
        optimizer.zero_grad()
        final_actv, *_ = model(*inputs)
                
        # segmentation loss(es)
        seg_actv = final_actv
        seg_target = target.squeeze(1)
        # main supervision
        seg_loss = seg_criterion(seg_actv, seg_target)
        total_loss = total_loss + seg_loss

        # backward pass
        total_loss.backward()
        optimizer.step()

        # store current losses
        total_losses.append(total_loss.item())
        seg_losses.append(seg_loss.item())
        
        running_loss += total_loss.item() 
        # print running loss
        if batch_idx % dump_period == dump_period - 1: 
            # this is an approximation because each patch has a different number of valid pixels
            progress_bar.set_postfix(loss=running_loss/dump_period)
            running_loss = 0.0
            
    # average losses over the epoch
    avg_total_loss = np.mean(total_losses, axis = 0)
    avg_seg_loss = np.mean(seg_losses, axis = 0)
    
    return avg_total_loss, avg_seg_loss

def fit_temp(model, 
             device, 
             dataloader, 
             optimizer, 
             n_batches, 
             seg_criterion, 
             temp_criterion=None, 
             lambda_temp=1.0,
             temp_align_criterion=None, 
             lambda_temp_align=0.0,
             update_period=1,
             seg_eval_year=2020):
    """
    Runs 1 training epoch, with auxiliary targets.
    n_batches is used to setup the progress bar only. The actual number of batches is defined by the dataloader.
    Returns average losses over the batches. Warning: the average assumes the same number of pixels contributed to the
    loss for each batch (unweighted average over batches)
    """
    model.train()
            
    seg_losses = []
    temp_losses = []
    temp_align_losses = []
    total_losses = []
    running_loss = 0.0
    dump_period = 25 # period (in number of batches) for printing running loss


    # train batch by batch
    
    progress_bar = tqdm(enumerate(dataloader), total=n_batches)
    optimizer.zero_grad()
    
    for batch_idx, data in progress_bar:      

        total_loss = torch.tensor([0.], requires_grad=True, device=device)
        total_weight = torch.tensor([0.], requires_grad=True, device=device)
        (temp_inputs, static_inputs, years), target = data
        
        
        temp_inputs = [d.to(device) for d in temp_inputs]
        static_inputs = static_inputs.to(device)
        years = years.squeeze(1).to(device)
        target = target.to(device) 

        # collect outputs of forward pass
        final_actv = model(temp_inputs, static_inputs, years=years) 
                
        # segmentation loss (only last time step)
        patch_idx_arr, t_idx_arr = torch.nonzero(years == seg_eval_year, as_tuple=True)
        seg_actv = torch.zeros_like(final_actv[..., 0, :, :])
        for (patch_idx, t_idx) in zip(patch_idx_arr, t_idx_arr): 
            seg_actv[patch_idx] = final_actv[patch_idx, :, t_idx]
        seg_target = target.squeeze(1)
        # main supervision
        seg_loss = seg_criterion(seg_actv, seg_target)
        total_loss = total_loss + seg_loss
        total_weight = total_weight + 1.
        
        if temp_criterion is not None:
            temp_loss, r_bootstrap = temp_criterion(final_actv, years)
            total_loss = total_loss + lambda_temp * temp_loss
            total_weight = total_weight + lambda_temp
            
        if temp_align_criterion is not None:
            temp_align_loss, _ = temp_align_criterion(final_actv, years)
            total_loss = total_loss + lambda_temp_align * temp_align_loss 
            total_weight = total_weight + lambda_temp_align

        # store current losses
        total_losses.append(total_loss.item())
        seg_losses.append(seg_loss.item())

        if temp_criterion is not None:
            temp_losses.append(temp_loss.item())
        if temp_align_criterion is not None:
            temp_align_losses.append(temp_align_loss.item())
        
        running_loss += total_loss.item() 
        # print running loss
        if batch_idx % dump_period == dump_period - 1: 
            # this is an approximation because each patch has a different number of valid pixels
            progress_bar.set_postfix(loss=running_loss/dump_period)
            running_loss = 0.0
        
        # divide by update_period so that the accumulated loss is equivalent to the case update_period = 1
        (total_loss / total_weight / update_period).backward()
            
        # backward pass
        if (batch_idx + 1) % update_period == 0:
            optimizer.step() 
            optimizer.zero_grad()
                    
    if (batch_idx + 1) % update_period != 0:
        # parameter update for remaining batches
        optimizer.step() 

    # average losses along the batch
    avg_total_loss = np.mean(total_losses)
    avg_seg_loss = np.mean(seg_losses) 
    
    if temp_criterion is None:
        avg_temp_loss = NaN  
    else:
        avg_temp_loss = np.mean(temp_losses)
    if temp_align_criterion is None:
        avg_temp_align_loss = NaN  
    else:
        avg_temp_align_loss = np.mean(temp_align_losses)
    
    return avg_total_loss, avg_seg_loss, avg_temp_loss, avg_temp_align_loss

    
class MyCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=I_NODATA_VAL, weight=None):
        super().__init__(ignore_index=ignore_index, weight=weight, reduction='none')
    
    def forward(self, pred, target, keep_time=False, return_count=False):
        loss = super().forward(pred, target)
        mask = target!=self.ignore_index
        if not keep_time:
            reduced_loss = torch.mean(loss[mask])
            if not return_count:
                return reduced_loss.nan_to_num()
            else:
                return reduced_loss.nan_to_num(), torch.sum(mask)
        else:
            count = torch.sum(mask, dim=(0, 2, 3))
            reduced_loss = torch.sum(loss * mask.float(), dim=(0, 2, 3)) / count
            if not return_count:
                return reduced_loss.nan_to_num()
            else:
                return reduced_loss.nan_to_num(), count


class MyBCELoss(nn.BCELoss): 
    """Converts the target to float internally"""
    def __init__(self, *args, ignore_val=255, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_val = ignore_val
        
    def forward(self, pred, target):
        if self.ignore_val is not None:
            mask = target!=self.ignore_val
        if pred.ndim == 4:
            pred = pred.squeeze(1) # squeeze the channel dimension
        if self.ignore_val is None:
            target = target.float()
            loss = super().forward(pred, target)
        elif torch.any(mask):
            target = target.float()
            loss = super().forward(pred[mask], target[mask]) # this flattens the tensors
        else:
            loss = pred.new_zeros((1,))
        return loss         
    
class MyBCEWithLogitsLoss(nn.BCEWithLogitsLoss): 
    """Converts the target to float internally"""
    def __init__(self, *args, ignore_val=255, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_val = ignore_val
        
    def forward(self, pred, target):
        if self.ignore_val is not None:
            mask = target!=self.ignore_val
        if pred.ndim == 4:
            pred = pred.squeeze(1) # squeeze the channel dimension
        if self.ignore_val is None:
            target = target.float()
            loss = super().forward(pred, target)
        elif torch.any(mask):
            target = target.float()
            loss = super().forward(pred[mask], target[mask])
        else:
            loss = pred.new_zeros((1,))
        return loss
    
class MyTemporalConsistencyLoss(nn.Module):
    def __init__(self, 
                 ignore_val=255, 
                 seg_normalization=None, 
                 use_temp_weights=False, 
                 bootstrap_beta=0,
                 bootstrap_threshold=0.5):
        """ 
        beta: beta parameter for bootstrapping (inspired by 
            "Training Deep Neural Networks on Noisy Labels with Bootstrapping", Reed et al., 
            https://arxiv.org/abs/1412.6596)
        """
        super().__init__()
        self.ignore_val = ignore_val 
        self.bootstrap_beta = bootstrap_beta
        self.bootstrap_threshold = bootstrap_threshold
        
        if seg_normalization is None:
            self.seg_normalization = nn.Softmax(dim = 1) 
        else:
            self.seg_normalization = seg_normalization
        if use_temp_weights:
            self.temp_average = self.weighted_temporal_average
        else:
            self.temp_average = self.unweighted_temporal_average
            
            
    def unweighted_temporal_average(self, loss, *args, **kwargs):
        return torch.mean(loss, dim=-1), None
    
    def get_temporal_average_weights(self, years):
        # time should be the last dimension
        # assumes no gaps in between years
        weights = 1. / (years[..., 1:] - years[..., :-1]) 
        weights[years[..., 1:] == self.ignore_val] = 0 
        weights[years[..., :-1] == self.ignore_val] = 0
        weights[torch.isnan(weights)] = 0
        return weights
    
    def weighted_temporal_average(self, loss, weights=None, years=None):
        if weights is None:
            if years is None:
                raise ValueError('"years" variable is necessary to compute weights')
            else:
                weights = self.get_temporal_average_weights(years)
        loss = torch.sum(loss * weights, dim=-1) / torch.sum(weights, dim=-1)
        return loss, weights
        
    def compute_loss_terms(self, pred_t0, pred_t1):
        return NotImplementedError
    
    def compute_loss(self, pred, years=None, weights=None, return_per_year=False):
        # dimensions can be (batch, time, h, w) or (batch, class, time, h, w)
        pred_t0, pred_t1 = pred[..., :-1, :, :], pred[..., 1:, :, :] # there should be no time gaps in the predictions
        if self.bootstrap_beta > 0 and self.bootstrap_threshold < 1: 
            pseudo_labels = torch.clone(pred_t0)
            diff = pred_t0 - pred_t1
            # bootstrap when forest loss is detected
            mask = diff > self.bootstrap_threshold
            r_bootstrap = torch.mean(mask.float()) # ratio of bootstrapped pixels
            if torch.any(mask):
                pseudo_labels[mask] = self.bootstrap_beta * pred_t1[mask] + (1-self.bootstrap_beta) * pred_t0[mask]
            per_year_loss = self.compute_loss_terms(pseudo_labels, pred_t1)
        else:
            r_bootstrap = 0
            per_year_loss = self.compute_loss_terms(pred_t0, pred_t1)
        loss, weights = self.temp_average(per_year_loss, weights, years)
        avg_loss = torch.mean(loss, dim=0)
        if return_per_year:
            # loss terms along a column do not correspond to the same year
            # cast the loss terms in a dict with year-specific keys
            years_after = years[:, 1:]
            unique_years = torch.unique(years_after[~torch.isnan(years_after)])
            avg_per_year_loss = {}
            for y in unique_years:
                avg_per_year_loss[y.item()] = torch.mean(per_year_loss[years_after == y]).detach().cpu()
            return (avg_loss, avg_per_year_loss), weights, r_bootstrap
        else:
            return avg_loss, weights, r_bootstrap
    
    def forward(self, pred, years=None, return_per_year=False):
        """
        When 'years' is None the loss is not reduced
        """
        pred = self.seg_normalization(pred) # batch_size x n_classes x n_t x h x w
        loss, weights, r_bootstrap = self.compute_loss(pred, years=years, return_per_year=return_per_year)
        return loss, r_bootstrap               

    
class MyTemporalMSELoss(MyTemporalConsistencyLoss):
    def __init__(self, ignore_val=255, seg_normalization=None, 
                 use_temp_weights=False, **kwargs):
        super().__init__(ignore_val=ignore_val, 
                         seg_normalization=seg_normalization, 
                         use_temp_weights=use_temp_weights,
                         **kwargs)
        
    def compute_loss_terms(self, pred_t0, pred_t1):
        loss = (pred_t1 - pred_t0)**2
        if loss.ndim == 4: # no "class" dimension
            loss = loss.mean(dim=(-2, -1))
        else:
            loss = loss.mean(dim=(1, -2, -1))
        return loss
    
class MyTemporalCosineSimilarityLoss(MyTemporalConsistencyLoss):
    """
    Cosine similarity loss on spatial gradients
    """
    def __init__(self, 
                 device, 
                 kernel_size=7, 
                 ignore_val=255, 
                 seg_normalization=None, 
                 use_temp_weights=False, 
                 **kwargs):
        """ kernel_size must be odd """
        super().__init__(ignore_val=ignore_val, 
                         seg_normalization=seg_normalization, 
                         use_temp_weights=use_temp_weights,
                         **kwargs)
        self.hgrad = nn.Conv2d(in_channels=1, 
                               out_channels=1, 
                               kernel_size=kernel_size, 
                               bias=False,
                               padding=0)
        # inputs of the conv will have size (batch_size, n_classes, n_t, H, W)
        self.hgrad.weight.requires_grad = False
        self.grad_margin = (kernel_size-1)//2
        idx_range = torch.arange(-(kernel_size-1)//2, kernel_size//2 + 1)
        m_i, m_j = torch.meshgrid(idx_range, idx_range, indexing='ij')
        hgrad_weight = m_j / (m_i**2 + m_j**2)
        hgrad_weight[torch.isnan(hgrad_weight)] = 0.0
        self.hgrad.weight = torch.nn.Parameter(hgrad_weight.float().unsqueeze(0).unsqueeze(0).to(device))
        
        self.vgrad = nn.Conv2d(in_channels=1, 
                               out_channels=1, 
                               kernel_size=kernel_size, 
                               bias=False,
                               padding=0)
        self.vgrad.weight.requires_grad = False
        vgrad_weight = m_i / (m_i**2 + m_j**2)
        vgrad_weight[torch.isnan(vgrad_weight)] = 0.0
        self.vgrad.weight = torch.nn.Parameter(vgrad_weight.float().unsqueeze(0).unsqueeze(0).to(device))
        self.cos_sim = nn.CosineSimilarity(dim=-3)
        
    def get_2d_grad(self, t):
        t_hgrad = self.hgrad(torch.flatten(t.float(), start_dim=0, end_dim=-3).unsqueeze(-3))
        t_hgrad = t_hgrad.reshape((*t.shape[:-2], 1, *t_hgrad.shape[-2:]))
        
        t_vgrad = self.vgrad(torch.flatten(t.float(), start_dim=0, end_dim=-3).unsqueeze(-3))
        t_vgrad = t_vgrad.reshape((*t.shape[:-2], 1, *t_vgrad.shape[-2:]))
        return torch.cat((t_hgrad, t_vgrad), dim=-3)
        
    def compute_loss_terms(self, pred_t0, pred_t1):
        pred_t0_grad = self.get_2d_grad(pred_t0)
        pred_t1_grad = self.get_2d_grad(pred_t1)
        loss = torch.mean(1 - self.cos_sim(pred_t0_grad, pred_t1_grad), dim=(1, -2, -1)) 
        return loss
    
class MyGradDotTemporalLoss(MyTemporalCosineSimilarityLoss):
    """
    Equivalent to weighting 1-cos_sim(x0, x1) with 
        - sqrt(norm2(x0)*norm2(x1)), for the symmetrical version
        - norm2(x0), for the asymmetrical version
    Encourages large spatial gradients to align between consecutive time steps
    Sobel filter calculation from 
    https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size/41065243#41065243
    """
    def __init__(self, 
                 device, 
                 kernel_size=7, 
                 ignore_val=255, 
                 eps=1e-8, 
                 seg_normalization=None, 
                 use_temp_weights=False, 
                 scale_by_norm=True, # set to False for ablation study
                 asymmetrical=False,
                 **kwargs):
        super().__init__(device, 
                         kernel_size=kernel_size, 
                         ignore_val=ignore_val, 
                         seg_normalization=seg_normalization, 
                         use_temp_weights=use_temp_weights,
                         **kwargs)
        self.eps = eps
        self.scale_by_norm = scale_by_norm
        self.asymmetrical = asymmetrical * scale_by_norm
        
    def compute_loss_terms(self, pred_t0, pred_t1):
        pred_t0_grad = self.get_2d_grad(pred_t0)
        pred_t1_grad = self.get_2d_grad(pred_t1)
        
        # compute dot product between gradients at t0 and t1
        pred_graddot = torch.einsum('...ijk,...ijk->...jk', pred_t0_grad, pred_t1_grad)
        
        # compute norm of gradients
        norm_t0 = torch.sqrt(torch.sum(pred_t0_grad**2, dim=-3) + self.eps)
        norm_t1 = torch.sqrt(torch.sum(pred_t1_grad**2, dim=-3) + self.eps)
        if self.scale_by_norm:
            if self.asymmetrical:
                a = norm_t1
                loss = norm_t1 - pred_graddot / (norm_t0 + self.eps)
            else:
                a = torch.sqrt(norm_t0 * norm_t1 + self.eps)
                loss = a - pred_graddot / a
        else:
            loss = 1 - pred_graddot / (norm_t0 * norm_t1 + self.eps)
        
        # do not sum along batch dimension because pair-wise weights might be different
        if loss.ndim == 4: # no "class" dimension
            if self.scale_by_norm:
                loss = loss.sum(dim=(-2, -1)) / a.sum(dim=(-2, -1))
            else:
                loss = loss.mean(dim=(-2, -1))
        else:
            if self.scale_by_norm:
                loss = loss.sum(dim=(1, -2, -1)) / a.sum(dim=(1, -2, -1)) 
            else:
                loss = loss.mean(dim=(1, -2, -1))
        return loss
    
class MyGradNormTemporalLoss(MyTemporalCosineSimilarityLoss):
    """
    For ablation study of MyGradDotTemporalLoss
    """
    def __init__(self, 
                device, 
                kernel_size=7, 
                ignore_val=255, 
                eps=1e-8, 
                seg_normalization=None, 
                use_temp_weights=False, 
                **kwargs):
        super().__init__(device, 
                        kernel_size=kernel_size, 
                        ignore_val=ignore_val, 
                        seg_normalization=seg_normalization, 
                        use_temp_weights=use_temp_weights,
                        **kwargs)
        self.eps = eps
        
    def compute_loss_terms(self, pred_t0, pred_t1):
        pred_t1_grad = self.get_2d_grad(pred_t1)
        
        norm_t1 = torch.sqrt(torch.sum(pred_t1_grad**2, dim=-3) + self.eps)
        loss = norm_t1

        # do not sum along batch dimension because pair-wise weights might be different
        if loss.ndim == 4: # no "class" dimension
            loss = loss.mean(dim=(-2, -1))
        else:
            loss = loss.mean(dim=(1, -2, -1))
        return loss
    
class MyTemporalCELoss(MyTemporalConsistencyLoss):
    def __init__(self, decision_func, seg_criterion, ignore_val=255, seg_normalization=None, 
                 use_temp_weights=False, **kwargs):
        super().__init__(ignore_val=ignore_val, 
                         seg_normalization=seg_normalization, 
                         use_temp_weights=use_temp_weights,
                         **kwargs)
        
        self.decision_func = decision_func
        self.seg_criterion = seg_criterion
        # reduction and nodata values are handled by the parent class
        self.seg_criterion.reduction = 'none'
        self.seg_criterion.ignore_val = None
        
    def compute_loss_terms(self, pred_t0, pred_t1):
        
        labels_t0 = self.decision_func(pred_t0)
        labels_t1 = self.decision_func(pred_t1)
        loss = (self.seg_criterion(pred_t0, labels_t1) + self.seg_criterion(pred_t1, labels_t0)) / 2
        if loss.ndim == 4: # no "class" dimension
            loss = loss.mean(dim=(-2, -1))
        else:
            loss = loss.mean(dim=(1, -2, -1))
        return loss
        
        
        
        