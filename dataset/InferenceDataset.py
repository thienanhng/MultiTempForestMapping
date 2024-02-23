import numpy as np
import torch
import rasterio
import rasterio.merge
import rasterio.transform
import rasterio.warp
from rasterio.windows import Window
from rasterio.enums import Resampling
import random
from torch.utils.data.dataset import Dataset
from scipy.ndimage import binary_dilation
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'Data')

class InferenceDataset(Dataset):

    """
    Dataset for inference. 
    - Generates regularly spaced, and possibly overlapping patches over a dataset. 
    - Reads each tile with a margin around it
    - Supports multi-source input and target rasters with different depths
    """

    def __init__(self, 
                 input_fns, 
                 input_vrt_fns, 
                 n_tiles, 
                 exp_utils, 
                 batch_size=16, 
                 patch_size=128, 
                 padding=64, 
                 target_fns=None, 
                 target_vrt_fns=None, 
                 input_nodata_val=None, 
                 target_nodata_val=None,
                 *args,
                 **kwargs):
        """
        Args:
            - input_fns (dataframe): input filename for each tile and each input source
            - input_vrt_fns (dict of str): virtual mosaic filename for each input source
            - target_fns (dataframe): target filename for each tile and each input source (with NaN cells for 
                non-available target tiles)
            - target_vrt_fns (dict of str): virtual mosaic filename for the ground truth
            - exp_utils (ExpUtils): object containing the information about the data sources and the patch 
                parameters (patch size, patch stride)
        """
        
        # set parameters
        self.n_inputs = len(input_vrt_fns)
        self.input_fns = input_fns
        self.input_vrt = self._open_vrt(input_vrt_fns)
        self.n_tiles = n_tiles

        if target_vrt_fns is None:
            self.sample_target = False
            self.target_vrt = None
            self.target_fns = None
        else:
            self.sample_target = True
            self.target_fns = target_fns
            self.target_vrt = self._open_vrt(target_vrt_fns)

        self.exp_utils = exp_utils
        self.patch_size = patch_size 
        self.patch_stride = patch_size - 2 * padding 
        self.tile_margin = padding

        self.batch_size = batch_size
        
        self.input_nodata_val = input_nodata_val
        self.target_nodata_val = target_nodata_val
                
        # for segmentation evaluation around object boundaries
        r = 25 # buffer around target boundaries (in pixels)
        y, x = np.ogrid[-r: r+1, -r: r+1]
        self.dilation_mask = x**2+y**2 <= r**2
        
    @staticmethod   
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _open_vrt(self, vrt_fns):
        vrt_dict = {k: rasterio.open(fn, 'r') for k, fn in vrt_fns.items()}
        
        return vrt_dict
        
    def _get_patch_coordinates(self, height, width): 
        """
        Fills self.patch_coordinates with an array of dimension (n_patches, 2) containing upper left pixels of patches, 
        at 1m resolution
        """
        height, width = int(height), int(width)
        xs = list(range(0, height - self.patch_size, self.patch_stride)) + [height - self.patch_size]
        ys = list(range(0, width - self.patch_size, self.patch_stride)) + [width - self.patch_size]
        xgrid, ygrid = np.meshgrid(xs, ys)
        return np.vstack((xgrid.ravel(), ygrid.ravel())).T
    
    def _get_input_nodata_mask(self, data_dict, height, width): 
        """
        Create nodata mask. A nodata pixel in the mask corresponds to an overlapping nodata pixel in any of the inputs.
        The mask also covers the margins.
        """
        check = np.full((height, width), False) 
        
        # check each input
        for k in data_dict:
            op1, _ = self.exp_utils.input_nodata_check_operator[k]  
            if self.input_nodata_val[k] is not None:
                check_im = op1(data_dict[k] == self.input_nodata_val[k], axis = -1) 
                check = np.logical_or(check, check_im[:height, :width]) 
        return check

    def _read_image_file(self, vrt, fn, max_margin=0, squeeze=True, res=1.0):
        """Resamples to 1m by default"""
        with rasterio.open(os.path.join(DATA_DIR, fn), 'r') as f_tile:
            left, top = f_tile.bounds.left, f_tile.bounds.top
            h, w, = f_tile.height, f_tile.width
            orig_res = f_tile.res[0]
            count = f_tile.count
        i_min, j_min = vrt.index(left, top)
        if max_margin > 0:
            # compute available margins around the tile
            max_margin = max_margin // orig_res
            top_margin = min(max(0, i_min-max_margin), max_margin)
            left_margin = min(max(0, j_min-max_margin), max_margin)
            bottom_margin = min(max(0, vrt.height - (i_min + h+max_margin)), max_margin)
            right_margin = min(max(0, vrt.width - (j_min + w+max_margin)), max_margin)
        else:
            top_margin, left_margin, bottom_margin, right_margin = 0, 0, 0, 0
        # read the tile + margins
        if res is None:
            res = orig_res
        res_ratio = orig_res/res
        total_w_orig = w + left_margin + right_margin
        total_h_orig = h + top_margin + bottom_margin
        win = Window(   j_min - left_margin, 
                        i_min - top_margin, 
                        total_w_orig, 
                        total_h_orig)
        total_w_out = int(total_w_orig * res_ratio)
        total_h_out = int(total_h_orig * res_ratio)
        out_shape = (count,
                     total_h_out,
                     total_w_out)
        data = vrt.read(window=win, out_shape=out_shape, resampling=Resampling.bilinear)
        
        # check if the tile is empty
        if np.all(data == vrt.nodata):
            return None, None, None
            
        if data.shape[0] == 1 and squeeze:
            data = data.squeeze(0)
        else:
            data = np.moveaxis(data, (1, 2, 0), (0, 1, 2))
            
        return data, (total_h_out, total_w_out), \
                (top_margin * res_ratio, 
                 left_margin * res_ratio, 
                 bottom_margin * res_ratio, 
                 right_margin * res_ratio)          
                       
    def _get_masked_target(self, target_tile, nodata_val):
        """assign all pixels far from class borders to 255, excluding borders corresponding to nodata values"""
        # comparison with pixel on the right
        right_diff = (target_tile[:-1, :-1] != target_tile[:-1, 1:]) \
                    * (target_tile[:-1, :-1] != nodata_val) \
                    * (target_tile[:-1, 1:] != nodata_val)
        # comparison with pixel below
        below_diff = (target_tile[:-1, :-1] != target_tile[1:, :-1]) \
                    * (target_tile[:-1, :-1] != nodata_val) \
                    * (target_tile[1:, :-1] != nodata_val)
        borders = np.logical_or(right_diff, below_diff)
        dilated_borders = binary_dilation(borders, self.dilation_mask)
        dilated_borders = np.pad(dilated_borders, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        target_borders_tile = np.full_like(target_tile, fill_value=nodata_val) * (1-dilated_borders) + \
                                target_tile * dilated_borders
        return target_borders_tile
    
    def _build_batches(self, data, coords_per_batch):
        """
        First dimension of data should be the depth (channels)
        Output: list of batches
        """
        num_batches = len(coords_per_batch)
        batches = [None] * num_batches
        mult_channels = data.dim() > 2
        if mult_channels:
            shape = (self.patch_size, self.patch_size, data.shape[0])
            data_copy = torch.clone(data).movedim((1, 2, 0),(0, 1, 2)) #to make the indexing easier
        else:
            shape = (self.patch_size, self.patch_size)
            data_copy = torch.clone(data)
        for batch_num in range(num_batches):
            this_batch_size = coords_per_batch[batch_num].shape[0]
            batch = torch.empty((this_batch_size, *shape), dtype = data.dtype)
            for patch_num in range(this_batch_size):
                xp, yp = coords_per_batch[batch_num][patch_num]
                x_start, x_stop = xp, xp +self.patch_size
                y_start, y_stop = yp, yp + self.patch_size
                batch[patch_num] = data_copy[x_start:x_stop, y_start:y_stop]
            if mult_channels:
                batch = batch.movedim((0, 3, 1, 2), (0, 1, 2, 3))
            batches[batch_num] = batch
        return batches
    
    def _check_tile_size(self, key, dest_dims, source_dims, data, nodataval, fn=None):
        dest_height, dest_width, dest_margins = dest_dims
        source_height, source_width, source_margins = source_dims
        # if the margins are different, pad/crop the current image
        if source_margins != dest_margins: 
            pad = [int(tm - m) for tm, m in zip(dest_margins, source_margins)]
            if any([p < 0 for p in pad]):
                crop = [-p if p<0 else 0 for p in pad]
                data = data[crop[0]:data.shape[0]-crop[2], crop[1]:data.shape[1]-crop[3]]
                pad = [p if p > 0 else 0 for p in pad]
            if any([p > 0 for p in pad]):
                data = np.pad(data, 
                                ((pad[0], pad[2]), (pad[1], pad[3]), (0,0)), 
                                mode='constant', 
                                constant_values=nodataval)
            source_height, source_width = data.shape[0], data.shape[1]
        # the main part of the image has different dimensions
        if source_height != dest_height or source_width != dest_width: 
            if fn is not None:
                print(fn)
            print('The dimensions of the images do not match: '
                                '\n\t(height={}, width={}, margins=({})) for the previous image v.s'
                                '\n\t(height={}, width={}, margins=({})) for {}.'
                                .format(dest_height, dest_width, ', '.join([str(m) for m in dest_margins]), 
                                        source_height, source_width, ', '.join([str(m) for m in source_margins]), key))
            if source_height < dest_height or source_width < dest_width:
                # this assumes top left corners are aligned
                if source_height < dest_height:
                    gap = int(dest_height - source_height)
                    print('Padding with {} row at the bottom.'.format(gap))
                    data = np.pad(data, 
                                ((0, gap), (0, 0), (0, 0)), 
                                mode='edge', #'constant', 
                    )#constant_values=nodataval)
                if source_width < dest_width:
                    gap = int(dest_width - source_width)
                    print('Padding with {} column at the right.'.format(gap))
                    data = np.pad(data, 
                                ((0, 0), (0, gap), (0, 0)), 
                                mode='edge', #constant', 
                    )#constant_values=nodataval)
            else:
                return None
        return data
    
    def _read_all_tile_data(self, idx, data_names, vrt_dict, fns_dict, nodata_dict, tile_dims=None):
        image_data = {}
        if tile_dims is not None:
            tile_height, tile_width, tile_margins = tile_dims
        else:
            tile_height, tile_width, tile_margins = None, None, None
        for key in data_names:
            fn = fns_dict[key][idx]
            if pd.isna(fn):
                data = None
            else:
                data, dims, margins = self._read_image_file(vrt_dict[key], 
                                                                    fn, 
                                                                    max_margin=self.tile_margin, 
                                                                    squeeze = False)
                if data is None:
                    return (None,) * 5
                # check the size of the image
                current_height, current_width = dims
                current_margins = margins
                if tile_height is None:
                    tile_height, tile_width = current_height, current_width
                    tile_margins = current_margins
                else:
                    data = self._check_tile_size(key=key,
                                                dest_dims=(tile_height, tile_width, tile_margins),
                                                source_dims=(current_height, current_width, current_margins),
                                                data=data,
                                                nodataval=nodata_dict[key],
                                                fn=fn)
            image_data[key] = data
            
        return image_data, tile_height, tile_width, tile_margins

    def __getitem__(self, idx):
        '''
        Output:
            - data (list of tensors): contains patches corresponding to the same location in each input source 
            - np.array containing upper left coordinates of the patch
        '''

        #### read tiles
        input_image_data, tile_height, tile_width, tile_margins = self._read_all_tile_data(
                                                                                idx, 
                                                                                data_names=self.exp_utils.inputs, 
                                                                                vrt_dict=self.input_vrt,
                                                                                fns_dict=self.input_fns,
                                                                                nodata_dict=self.input_nodata_val,
                                                                                tile_dims = None)
        if input_image_data is None:
            print('Skipping tile.')
            return (None,) * 6
                    
        if self.sample_target:
            target_image_data, *_ = self._read_all_tile_data(idx, 
                                                            data_names=self.exp_utils.targets, 
                                                            vrt_dict=self.target_vrt,
                                                            fns_dict=self.target_fns,
                                                            nodata_dict=self.target_nodata_val,
                                                            tile_dims = (tile_height, tile_width, tile_margins))
        else:  
            target_image_data = None        
              
        #### build batches        
        input_nodata_mask = self._get_input_nodata_mask(input_image_data, tile_height, tile_width, tile_margins)
        coords = self._get_patch_coordinates(tile_height, tile_width)
        num_patches = coords.shape[0]
        num_batches, remainder = divmod(num_patches, self.batch_size)
        if remainder > 0:
            num_batches += 1
        #split the coordinates into chunks corresponding to the batches
        coords_per_batch = [coords[i:min(i+self.batch_size, num_patches)] for i in range(0, num_patches, self.batch_size)]
            
        # build input batches
        input_batches = {}
        input_image_data = self.exp_utils.preprocess_static_inputs(input_image_data, mode='eval')
        for key, data in input_image_data.items():
            input_batches[key] = self._build_batches(data, coords_per_batch)

        # build target batches
        target_batches = None
        target_tiles = None
        top_margin, left_margin, bottom_margin, right_margin = [int(m) for m in tile_margins]
        if self.sample_target:
            target_batches = {}
            target_tiles = {}
            for key in target_image_data: 
                if target_image_data[key] is not None:
                    target_data_for_batches = self.exp_utils.preprocess_training_target(target_image_data[key], key)
                    target_batches[key] = self._build_batches(target_data_for_batches, coords_per_batch)
                    target_tile = self.exp_utils.preprocess_inference_target(
                                    target_image_data[key][top_margin:tile_height-bottom_margin, 
                                                           left_margin:tile_width-right_margin],
                                    key) 
                    target_borders_tile = self._get_masked_target(target_tile, self.target_nodata_val[key])
                    target_tiles[key] = (target_tile, target_borders_tile)
            
            
        # remove margins in the nodata mask
        input_nodata_mask = input_nodata_mask[top_margin:tile_height-bottom_margin, 
                                              left_margin:tile_width-right_margin]
        
        return (input_batches, target_batches), \
                target_tiles, \
                coords_per_batch, (tile_height, tile_width), tile_margins, input_nodata_mask   

    def __len__(self):
        return self.n_tiles
    
    def __del__(self):
        
        for item in self.input_vrt.values(): 
            if isinstance(item, dict):
                for vrt in item.values():
                    vrt.close()
            else:
                item.close()
                
        if self.target_vrt is not None:
            for item in self.target_vrt.values(): 
                if isinstance(item, dict):
                    for vrt in item.values():
                        vrt.close()
                else:
                    item.close()
        
    
class TempInferenceDataset(InferenceDataset):
    
    def __init__(self, 
                 input_fns,
                 input_fns_per_year, 
                 input_vrt_fns, 
                 n_tiles, 
                 exp_utils, 
                 batch_size=16, 
                 patch_size=128, 
                 padding=4, 
                 target_fns=None,
                 target_vrt_fns_per_year=None, 
                 target_vrt_fns=None, 
                 input_nodata_val=None, 
                 target_nodata_val=None):
        
        '''
        Builds batches with patches of diverse time footprints (through compacting and padding along the time direction). 
        '''
        
        self.input_fns_per_year = input_fns_per_year
        self.target_fns_per_year = target_vrt_fns_per_year
        
        super().__init__(input_fns=input_fns,
                        input_vrt_fns=input_vrt_fns, 
                        n_tiles=n_tiles, 
                        exp_utils=exp_utils, 
                        batch_size=batch_size, 
                        patch_size=patch_size,
                        padding=padding,
                        target_fns=target_fns,
                        target_vrt_fns=target_vrt_fns, 
                        input_nodata_val=input_nodata_val, 
                        target_nodata_val=target_nodata_val)
        
        
        
    def _open_vrt(self, input_vrt_fns):
        vrt_dict = {}
        for k, item in input_vrt_fns.items():
            if isinstance(item, str):
                vrt = rasterio.open(item, 'r')
                vrt_dict[k] = vrt
            elif isinstance(item, dict):
                vrt_subdict = {}
                for key in item:
                    vrt = rasterio.open(item[key], 'r')
                    vrt_subdict[key] = vrt
                vrt_dict[k] = vrt_subdict
        
        return vrt_dict
    
    def _get_input_nodata_mask(self, data_dict, height, width, margins): 
        """
        Create nodata mask. A nodata pixel in the mask corresponds to an overlapping nodata pixel in any of the inputs.
        The mask also covers the margins.
        """
        # get the dimensions of the output
        n_steps = max([len(item) if isinstance(item, list) else 0 for item in data_dict.values()])
        check = np.full((n_steps, height, width), False) 
        tm, lm, bm, rm = [int(m) for m in margins]
        
        # check each input
        for k in data_dict:
            op1, _ = self.exp_utils.input_nodata_check_operator[k]  
            if self.input_nodata_val[k] is not None:
                nodata_val = self.input_nodata_val[k]
                if isinstance(data_dict[k], list): # temporal data
                    for j, image in enumerate(data_dict[k]):
                        # check along bands
                        im_height, im_width, _ = image.shape
                        check_im = op1(image == nodata_val, axis = -1)
                        # check if margins contain only nodata values
                        if tm > 0:
                            margin_data = image[:tm, lm:im_width-rm]
                            if np.all(margin_data == nodata_val):  
                                check_im[:2*tm] = True
                        if bm > 0:
                            margin_data = image[-bm:, lm:im_width-rm]
                            if np.all(margin_data == nodata_val):  
                                check_im[-2*bm:] = True
                        if lm > 0:
                            margin_data = image[tm:im_height-bm, :lm]
                            if np.all(margin_data == nodata_val):  
                                check_im[:, :2*lm] = True
                        if rm > 0:
                            margin_data = image[tm:im_height-bm, -rm:]
                            if np.all(margin_data == nodata_val):  
                                check_im[:, -2*rm:] = True
                        # check if margins corners contain only nodata values
                        if tm > 0 and lm > 0:
                            margin_data = image[:tm, :lm]
                            if np.all(margin_data == nodata_val):  
                                check_im[:2*tm, :2*lm] = True
                        if tm > 0 and rm > 0:
                            margin_data = image[:tm, -rm:]
                            if np.all(margin_data == nodata_val):  
                                check_im[:2*tm, -2*rm:] = True
                        if bm > 0 and lm > 0:
                            margin_data = image[-bm:, :lm]
                            if np.all(margin_data == nodata_val):  
                                check_im[-2*bm:, :2*lm] = True
                        if bm > 0 and rm > 0:
                            margin_data = image[-bm:, -rm:]
                            if np.all(margin_data == nodata_val):  
                                check_im[-2*bm:, -2*rm:] = True
                        check[j] = check_im
                else: # static data
                    # check along bands
                    image = data_dict[k]
                    im_height, im_width, _ = image.shape
                    check_im = op1(image == nodata_val, axis = -1) 

                    # check if margins contain only nodata values
                    if tm > 0:
                        margin_data = image[:tm, lm:im_width-rm]
                        if np.all(margin_data == nodata_val):  
                            check_im[:2*tm] = True
                    if bm > 0:
                        margin_data = image[-bm:, lm:im_width-rm]
                        if np.all(margin_data == nodata_val):  
                            check_im[-2*bm:] = True
                    if lm > 0:
                        margin_data = image[tm:im_height-bm, :lm]
                        if np.all(margin_data == nodata_val):  
                            check_im[:, :2*lm] = True
                    if rm > 0:
                        margin_data = image[tm:im_height-bm, -rm:]
                        if np.all(margin_data == nodata_val):  
                            check_im[:, -2*rm:] = True
                    # check if margins corners contain only nodata values
                    if tm > 0 and lm > 0:
                        margin_data = image[:tm, :lm]
                        if np.all(margin_data == nodata_val):  
                            check_im[:2*tm, :2*lm] = True
                    if tm > 0 and rm > 0:
                        margin_data = image[:tm, -rm:]
                        if np.all(margin_data == nodata_val):  
                            check_im[:2*tm, -2*rm:] = True
                    if bm > 0 and lm > 0:
                        margin_data = image[-bm:, :lm]
                        if np.all(margin_data == nodata_val):  
                            check_im[-2*bm:, :2*lm] = True
                    if bm > 0 and rm > 0:
                        margin_data = image[-bm:, -rm:]
                        if np.all(margin_data == nodata_val):  
                            check_im[-2*bm:, -2*rm:] = True
                    check = np.logical_or(check, check_im) 
        return check
    
    def _read_all_tile_data(self, idx, data_names, vrt_dict, fns_per_year_dict, fns_dict, nodata_dict,
                            tile_dims=None):

        image_data = {}
        image_years = {}
        if tile_dims is not None:
            tile_height, tile_width, tile_margins = tile_dims
        else:
            tile_height, tile_width, tile_margins = None, None, None
        for key in data_names:
            if isinstance(vrt_dict[key], dict): # time series
                if fns_per_year_dict[key].loc[idx].isna().all():
                    data = None
                else:
                    data = []
                    years = []
                    margins = []
                    
                    # footprint_mismatch = False
                    for key_year in vrt_dict[key]: 
                        fn = fns_per_year_dict[key].loc[idx,key_year]
                        if not isinstance(fn, float): #fn is not nan
                            data_year, dims, margins_year = self._read_image_file(
                                                                    vrt_dict[key][key_year], 
                                                                    fn,
                                                                    max_margin=self.tile_margin,
                                                                    squeeze = False) 
                            if data_year is None:
                                continue
                            else:
                                # check the size of the image
                                current_height, current_width = dims 
                                current_margins = margins_year
                                if tile_height is None:
                                    tile_height, tile_width = current_height, current_width
                                    tile_margins = current_margins
                                else:
                                    data_year = self._check_tile_size(key=key,
                                                        dest_dims=(tile_height, tile_width, tile_margins),
                                                        source_dims=(current_height, current_width, current_margins),
                                                        data=data_year,
                                                        nodataval=nodata_dict[key],
                                                        fn=fn)
                                    if data_year is None:
                                        return None
                                data.append(data_year)
                                margins.append(current_margins)
                                year_str = key_year.split('_')[-1]
                                years.append(year_str)
                    image_years[key] = years
                
            else: # static data
                fn = fns_dict[key][idx]
                if pd.isna(fn):
                    data =  None
                else:
                    data, dims, margins = self._read_image_file(vrt_dict[key], 
                                                                    fn, 
                                                                    max_margin=self.tile_margin,
                                                                    squeeze = False)
                    if data is None: # if the static data tile is empty, the whole tile is discarded
                        return (None,) * 6
                    # check the size of the image
                    current_height, current_width = dims
                    current_margins = margins
                    if tile_height is None:
                        tile_height, tile_width = current_height, current_width
                        tile_margins = current_margins
                    else:
                        data = self._check_tile_size(key=key,
                                                    dest_dims=(tile_height, tile_width, tile_margins),
                                                    source_dims=(current_height, current_width, current_margins),
                                                    data=data,
                                                    nodataval=nodata_dict[key],
                                                    fn=fn)
                        if data is None:
                            return None
                    if key == 'target_multitemp':
                        image_years[key] = self.exp_utils.year_extractor(fns_dict[key][idx])
            
            image_data[key] = data
        return image_data, image_years, tile_height, tile_width, tile_margins
    
    def _build_padded_batches(self, data, coords_per_batch, time_footprints_per_batch):
        """
        Dimensions of data should be time, channels, H, W
        Output: list of batches
        """
        n_t = len(data)
        n_t_valid = np.max(np.sum(~time_footprints_per_batch, axis=-1)) # max length of footprints
        
        common_n_bands = np.all([d_t.dim() == data[0].dim() for d_t in data[1:]])
        common_n_bands = common_n_bands * np.all([d_t.shape[0] == data[0].shape[0] for d_t in data[1:]])
        if not common_n_bands:
            raise RuntimeError('All the images in the input times series should have the same number of bands')
        
        mult_channels = data[0].dim() > 2 # assumes common_input_bands
        if mult_channels:
            shape = (data[0].shape[0], self.patch_size, self.patch_size)
        else:
            shape = (self.patch_size, self.patch_size)
        dtype = data[0].dtype
       
        num_batches = len(coords_per_batch)
        batches = [[torch.empty((coords_per_batch[batch_num].shape[0], *shape), 
                                dtype=dtype) for batch_num in range(num_batches)] for _ in range( n_t_valid)] 
        # batches = [[[]]] 
        
        for batch_num in range(num_batches):
            this_batch_size = coords_per_batch[batch_num].shape[0]
            for patch_num in range(this_batch_size):
                xp, yp = coords_per_batch[batch_num][patch_num]
                x_start, x_stop = xp, xp +self.patch_size
                y_start, y_stop = yp, yp + self.patch_size
                fp = time_footprints_per_batch[batch_num][patch_num]
                current_step = 0
                for t in range(n_t):
                    if not fp[t]: 
                        batches[current_step][batch_num][patch_num] = data[t][..., x_start:x_stop, y_start:y_stop] 
                        current_step += 1 # no data gaps, compress data to the left
        return batches
    
    def get_batches(self, idx):
        '''
        Build batches from a tile. The times series are compacted and padded to the right (no time gaps)
        ''' 
        #### read tiles
        try:
            input_image_data, input_image_years, tile_height, tile_width, tile_margins = \
                                                                                self._read_all_tile_data(
                                                                                idx, 
                                                                                data_names=self.exp_utils.inputs, 
                                                                                vrt_dict=self.input_vrt,
                                                                                fns_per_year_dict=self.input_fns_per_year,
                                                                                fns_dict=self.input_fns,
                                                                                nodata_dict=self.input_nodata_val,
                                                                                tile_dims = None)
        except TypeError: # problem in reading data
            # print ('Skipping tile.')
            return (None,) * 9
        if input_image_data is None:
            # print ('Skipping tile.')
            return (None,) * 9
        
        if self.sample_target:
            target_image_data, target_image_years, *_ = self._read_all_tile_data(idx, 
                                                            data_names=self.exp_utils.targets, 
                                                            vrt_dict=self.target_vrt,
                                                            fns_per_year_dict=self.target_fns_per_year,
                                                            fns_dict=self.target_fns,
                                                            nodata_dict=self.target_nodata_val,
                                                            tile_dims=(tile_height, tile_width, tile_margins))

        else:
            target_image_data = None
                    
        #### build batches        
        input_nodata_mask = self._get_input_nodata_mask(input_image_data, tile_height, tile_width, tile_margins)
        n_steps = len(input_nodata_mask)
        
        coords = self._get_patch_coordinates(tile_height, tile_width)
        valid_coords = []
        num_patches = coords.shape[0]
        footprints = []
        # ignore nodata values that covers less than 100m2 and less than 10% or the patch area
        max_nodata_area = min(100**2, self.patch_size**2 / 10)
        for j in range(num_patches):
            xp, yp = coords[j]
            x_start, x_stop = xp, xp+self.patch_size
            y_start, y_stop = yp, yp+self.patch_size
            nodata_patch = input_nodata_mask[:, x_start:x_stop, y_start:y_stop].reshape(n_steps, -1)
            if not np.all(nodata_patch):
                # some tests to avoid using np.unique (time-consuming)
                if np.all(~nodata_patch): # there are no nodata values
                    valid_coords.append(coords[j])
                    footprints.append(np.full(nodata_patch.shape[0], fill_value=False))
                elif np.all(np.all(nodata_patch == nodata_patch[:, 0:1], axis=1)): # same footprint for every pixel
                    valid_coords.append(coords[j])
                    footprints.append(nodata_patch[:, 0])
                elif np.all(np.sum(nodata_patch, axis=-1) < max_nodata_area): # the nodata values cover small areas
                    valid_coords.append(coords[j])
                    footprints.append(np.full(nodata_patch.shape[0], fill_value=False))
                else: # several footprints in the patch
                    patch_footprints, counts = np.unique(nodata_patch, axis=-1, return_counts=True)
                    # ignore footprints corresponding to less than max_nodata_area or contain no valid time steps
                    valid_footprints_mask = np.any(~patch_footprints, axis=0) * (counts > max_nodata_area)
                    n_valid_footprints = np.sum(valid_footprints_mask)
                    if n_valid_footprints > 0: 
                        patch_footprints = patch_footprints[:, valid_footprints_mask]
                        # define a common footprint over the patch
                        if n_valid_footprints > 1:
                            common_footprint = np.any(patch_footprints, axis=1)
                        else:
                            common_footprint = patch_footprints.squeeze(-1)
                        if not np.all(common_footprint): # the footprint corresponds to at least one valid timestep
                            valid_coords.append(coords[j])
                            footprints.append(common_footprint)
        if len(footprints) > 0:                                
            # check if some years are not used in the batches
            footprints = np.stack(footprints)
            years_not_used = np.all(footprints, axis=0)
            if np.any(years_not_used):
                years_used = ~years_not_used
                footprints = footprints[:, years_used]
                for key in input_image_data: 
                    if isinstance(input_image_data[key], list):
                        input_image_data[key] = [item for item, used in zip(input_image_data[key],years_used) if used]
                        new_year_list = [item for item, used in zip(input_image_years[key],years_used) if used]
                        input_image_years[key] = new_year_list
                input_nodata_mask = input_nodata_mask[years_used]
            
                if self.sample_target:  
                    to_del = []
                    for key in target_image_data:
                        if key in target_image_years:
                            if np.all([y not in new_year_list for y in target_image_years[key]]):
                                to_del.append(key)
                    target_image_data = {k: v for k, v in target_image_data.items() if k not in to_del}
                    target_image_years = {k: v for k, v in target_image_years.items() if k not in to_del}
                    
            # split the groups into batches if the size of the group is larger than the batch size
            time_footprints_per_batch = []
            coords_per_batch = []
            num_patches = len(coords)
            num_batches, remainder = divmod(num_patches, self.batch_size)
            if remainder > 0:
                num_batches += 1
            for j in range(num_batches):
                start = j*self.batch_size
                stop = min((j+1)*self.batch_size, num_patches)
                coords_per_batch.append(
                    np.column_stack(valid_coords[start:stop]).T)
                time_footprints_per_batch.append(footprints[start:stop])
            try:
                time_footprints_per_batch = np.stack(time_footprints_per_batch)
            except ValueError:
                pass
                    
            # build batches
            
            # build input batches
            input_batches = {}
            input_image_data = self.exp_utils.preprocess_mixed_inputs(input_image_data, input_image_years)
            for key, data in input_image_data.items(): 
                if isinstance(data, list): # multitemporal data
                    input_batches[key] = self._build_padded_batches(data, 
                                                                    coords_per_batch, 
                                                                    time_footprints_per_batch) 
                else: # static data
                    input_batches[key] = self._build_batches(data, coords_per_batch)
                        
            # build target batches
            target_batches = None
            target_tiles = None
            top_margin, left_margin, bottom_margin, right_margin = [int(m) for m in tile_margins]
            if self.sample_target:
                target_batches = {}
                target_tiles = {}
                for key, item in target_image_data.items(): 
                    if item is not None:
                        if isinstance(item, np.ndarray): # non temporal
                            # target for computing the loss
                            target_batches[key] = self._build_batches(self.exp_utils.preprocess_training_target(item, key), 
                                                                        coords_per_batch)
                            # targets for evaluation metrics
                            target_tile = self.exp_utils.preprocess_inference_target(
                                            item[top_margin:tile_height-bottom_margin, 
                                                left_margin:tile_width-right_margin],
                                            key) 
                            target_borders_tile = self._get_masked_target(target_tile, self.target_nodata_val[key])
                            target_tiles[key] = (target_tile, target_borders_tile)
                        else: # temporal
                            # suboptimal because the tiles are padded with a margin when reading, then the margin is removed 
                            # here
                            target_tiles[key] = {}
                            for y, data in zip(target_image_years[key], item):
                                # targets for evaluation metrics
                                target_tile = self.exp_utils.preprocess_inference_target(
                                                data[top_margin:tile_height-bottom_margin, 
                                                    left_margin:tile_width-right_margin],
                                                key) 
                                if key in ['target_tlm', 'target_multitemp']:
                                    target_borders_tile = self._get_masked_target(target_tile, self.target_nodata_val[key])
                                    target_tiles[key][y] = (target_tile, target_borders_tile)
                                else:
                                    target_tiles[key][y] = target_tile   
                                    
            # remove margins in the nodata mask
            input_nodata_mask = input_nodata_mask[:, 
                                                top_margin:tile_height-bottom_margin, 
                                                left_margin:tile_width-right_margin]
            
            # join year dicts together
            years_dic = input_image_years
            if self.sample_target:
                for k, v in target_image_years.items():
                    years_dic[k] = v
                    
            time_footprints_per_batch = torch.from_numpy(~time_footprints_per_batch)
            # year metadata for each batch and each patch
            t_all_years = self.exp_utils.preprocess_year_info(years_dic['input_main'])
            years_per_batch = torch.full(time_footprints_per_batch.shape, fill_value=float('nan')) 
            for batch_idx in range(time_footprints_per_batch.shape[0]):
                for patch_idx in range(time_footprints_per_batch.shape[1]):
                    fp = time_footprints_per_batch[batch_idx][patch_idx]
                    y = t_all_years[fp]
                    years_per_batch[batch_idx][patch_idx][:len(y)] = y
            
            return (input_batches, target_batches), target_tiles, coords_per_batch, \
                    time_footprints_per_batch, years_per_batch, (tile_height, tile_width), tile_margins, \
                    input_nodata_mask, years_dic
        else:
            return (None,) * 9        
        
    def __getitem__(self, idx):
        return self.get_batches(idx)