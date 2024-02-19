import os
import rasterio
from rasterio.windows import Window
import shutil
import random
from osgeo import gdal
import pandas as pd
import numpy as np
import torch
from .eval_utils import cm2rates, rates2metrics, my_confusion_matrix, map_vals_from_nested_dicts
from dataset import InferenceDataset, TempInferenceDataset
from utils.exp_utils import I_NODATA_VAL, F_NODATA_VAL, YEAR_EXTRACTOR
from .write_utils import Writer
import random
from tqdm import tqdm
from collections import defaultdict
from functools import partial
import wandb
from rasterio.enums import Resampling


class Inference():
    """
    Class to perform inference and evaluate predictions on a set of samples. If used for validation during training, 
    the class must be instantiated once before training and Inference.infer() can be called at each epoch.
    Virtual mosaics with all the tiles for each source are created so that the Dataset can sample patches that overlap 
    several neighboring tiles. If they exist, the nodata values of the inputs rasters are used to fill the gaps, 
    otherwise a new nodata value is introduced depending on the raster data type. When calling infer(), the criteria 
    ignore_index attributes are modified accordingly.
    """
    def __init__(self, model, file_list, exp_utils, output_dir=None, 
                        evaluate=True, save_hard=True, save_soft=True, 
                        batch_size=32, patch_size=128, padding=64,
                        num_workers=0, device=0, undersample=1,
                        random_seed=None, wandb_tracking=False):

        """
        Args:
            - model (nn.Module): model to perform inference with
            - file_list (str): csv file containing the files to perform inference on (1 sample per row)
            - exp_utils (ExpUtils): object containing information of the experiment/dataset
            - output_dir (str): directory where to write output files
            - evaluate (bool): whether to evaluate the predictions
            - save_hard (bool): whether to write hard predictions into image files
            - save_soft (bool): whether to write soft predictions into image files
            - batch_size (int): batch size
            - num_workers (int): number of workers to use for the DataLoader. Recommended value is 0 because the tiles
                are processed one by one
            - device (torch.device): device to use to perform inference 
            - undersample (int): undersampling factor to reduction the size of the dataset. Example: if undersample = 100, 
                1/100th of the dataset's samples will be randomly picked to perform inference on.
            - random_seed (int): random seed for Pytorch
        """

        self.evaluate = evaluate  
        self.save_hard = save_hard
        self.save_soft = save_soft
        self.model = model
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.padding = padding
        self.num_workers = num_workers
        self.device = device
        self.undersample = undersample
        self.exp_utils = exp_utils
        self.patch_size = patch_size
        self.input_vrt_fn = None # used to indicate that virtual raster mosaic(s) has not been created yet
        self.wandb_tracking = wandb_tracking
        
        g = torch.Generator()
        if random_seed is not None:
            g.manual_seed(random_seed)
        self.g = g

        self.n_input_sources = self.exp_utils.n_input_sources

        # create a temporary directory to save virtual raster mosaics
        self.tmp_dir = 'tmp'
        i = 0
        while os.path.isdir(self.tmp_dir):
            i += 1
            self.tmp_dir = 'tmp_{}'.format(i)
        os.mkdir(self.tmp_dir)           
 
        file_list_ext = os.path.splitext(file_list)[-1]
        if file_list_ext == '.csv':
            df = pd.read_csv(file_list)
        else:
            raise ValueError('file_list should be a csv file ("*.csv")')
        self.n_tiles = len(df)
        self._fns_per_position_df = df # self._fns_per_position_df should not be modified
        
        # create the column strings to read the dataframe
        self.input_col_names = self._check_col_names(df, self.exp_utils.inputs)
        if self.evaluate:
            self.target_col_names = self._check_col_names(df, self.exp_utils.targets)

        # Initialize dictionary containing cumulative confusion matrices
        self.cum_cms = {}
        if self.evaluate:
            self.cum_cms = {key: {'seg': {},'seg_contours': {}} for key in self.exp_utils.targets}

                    
    def _check_col_names(self, df, source_names):
        col_names = list(df)
        valid_names = []
        for name in source_names:
            if name in col_names:
                valid_names.append(name)
            else:
                raise KeyError('Could not find {} column in dataset csv file'.format(name))
        return valid_names
            
            
    def get_vrt(self, fns, data_name, nodata_val, check_dtype=None): 
        #fns = df[col_name]       
        vrt_fn = os.path.join(self.tmp_dir, '{}.vrt'.format(data_name))
        if nodata_val is None:
            # read the first tile just to know the data type:
            with rasterio.open(fns[0], 'r') as f_tile:
                dtype = f_tile.profile['dtype']
                if check_dtype is not None:
                    if dtype != check_dtype:
                        raise TypeError('Expected type {} for {}, found {} instead.'.format(check_dtype, 
                                                                                            data_name, 
                                                                                            dtype))
            if dtype == 'uint8':
                new_nodata_val = I_NODATA_VAL
            elif dtype.startswith('uint'):
                new_nodata_val = I_NODATA_VAL
                print('WARNING: nodata value for {} set to {}'.format(data_name, I_NODATA_VAL))
            else:
                # the min and max float32 values are not handled by GDAL, using value -1 instead
                new_nodata_val = F_NODATA_VAL
                print('WARNING: nodata value for {} set to {}'.format(data_name, F_NODATA_VAL)) 
        else:
            new_nodata_val = nodata_val
        gdal.BuildVRT(  vrt_fn,  
                        list(fns),
                        VRTNodata=new_nodata_val,
                        allowProjectionDifference=True)  
        return vrt_fn , new_nodata_val
    
    
    def _get_vrt_from_df(self, df):
        """Build virtual mosaic rasters from files listed in dataframe df"""
        print('Building input virtual mosaics for inference...')
        self.input_fns = {}
        self.input_vrt_fns = {}
        self.input_vrt_nodata_val = {}
        for input_name, col_name in zip(self.exp_utils.inputs, self.input_col_names):
            self.input_fns[input_name] = df[col_name]
            vrt_fn, new_nodata_val = self.get_vrt(self.input_fns[input_name], 
                                                  col_name, 
                                                  self.exp_utils.input_nodata_val[input_name])
            self.input_vrt_nodata_val[input_name] = new_nodata_val
            self.input_vrt_fns[input_name] = vrt_fn
            
        if self.evaluate:
            print('Building target virtual mosaics for inference...')  
            self.target_fns = {}
            self.target_vrt_fns = {}
            self.target_vrt_nodata_val = {}
            for target_name, col_name in zip(self.exp_utils.targets, self.target_col_names):
                self.target_fns[target_name] = df[col_name] 
                vrt_fn, new_nodata_val = self.get_vrt(self.target_fns[target_name], 
                                                      col_name, 
                                                      self.exp_utils.target_nodata_val[target_name])
                self.target_vrt_nodata_val[target_name] = new_nodata_val
                self.target_vrt_fns[target_name] = vrt_fn
        else:
            self.target_vrt_fns = None
            self.target_vrt_nodata_val = None

    def _select_samples(self):
        """Select samples to perform inference on"""
        # use a random subset of the data
        if self.undersample > 1:
            idx = sorted(random.sample(range(self.n_tiles), self.n_tiles//self.undersample))
            df = self._fns_per_position_df.iloc[idx]
            return df.reset_index(drop = True)
        else:
            return self._fns_per_position_df
    
    def _dict_select_batch_and_todevice(self, dict, batch_idx):
        out_dict = {}
        for k, data in dict.items():
            if isinstance(data[batch_idx], list):
                out_dict[k] = [item[batch_idx].to(self.device) for item in data]
            else:
                out_dict[k] = data[batch_idx].to(self.device)
        return out_dict

    def _reset_cm(self):
        """Reset the confusion matrix/matrices with zeros"""
        if self.evaluate:
            self.cum_cms = map_vals_from_nested_dicts(self.cum_cms, lambda cm: np.zeros_like(cm))
                

    def _get_decisions(self, actv, target_data, nodata_mask=None, eval_channel=None, thresh=0.5):
        """Obtain decisions from soft outputs (argmax) and update confusion matrix/matrices"""                     
        # define the outputs 
        output = actv.squeeze(0) # remove singleton dimension for a 2-class problem
        # compute hard predictions
        output_hard = self.exp_utils.decision_func(output, thresh=thresh)
                    
        # apply nodata value to invalid pixels (must be done before computing the confusion matrices)
        if nodata_mask is not None:
            output_hard[nodata_mask] = self.exp_utils.i_out_nodata_val

        ########## update confusion matrices #########
        # main task
        if self.evaluate:
            for key, item in target_data.items():
                if isinstance(item, tuple): # non-temporal
                    target_all, target_borders = item
                    pred = output_hard if eval_channel is None else output_hard[eval_channel[key]]
                    cm_all = my_confusion_matrix(target_all, 
                                                        pred,
                                                        self.exp_utils.n_classes)
                    cm_borders = my_confusion_matrix(target_borders, 
                                                    pred,
                                                    self.exp_utils.n_classes)
                    try:
                        self.cum_cms[key]['seg'] += cm_all
                        self.cum_cms[key]['seg_contours'] += cm_borders
                    except (KeyError, TypeError): # the entry does not exist yet
                        self.cum_cms[key]['seg'] = cm_all
                        self.cum_cms[key]['seg_contours'] = cm_borders

                else: # dict -> temporal target
                    for y, data in item.items():
                        target_all, target_borders = data
                        if eval_channel is None:
                            pred = output_hard
                        else:
                            try:
                                pred = output_hard[eval_channel[key][y]]
                            except KeyError as e:
                                print('Warning: {} available for {} but not found in input time series'.format(key, 
                                                                                                                   y))
                                continue
                        cm_all = my_confusion_matrix(target_all, 
                                                        pred,
                                                        self.exp_utils.n_classes)
                        cm_borders = my_confusion_matrix(target_borders, 
                                                    pred,
                                                    self.exp_utils.n_classes)
                        try:
                            self.cum_cms[key]['seg'][y] += cm_all
                            self.cum_cms[key]['seg_contours'][y] += cm_borders
                        except KeyError: # the entry does not exist yet
                            self.cum_cms[key]['seg'][y] = cm_all
                            self.cum_cms[key]['seg_contours'][y] = cm_borders
                            

        return output_hard

    def _compute_metrics(self):
        """Compute classification metrics from confusion matrices"""
        if 'target_multitemp' in self.cum_cms:
            self.cum_cms['target_multitemp'] = self._aggregate_year_cms(self.cum_cms['target_multitemp'])

        reports = map_vals_from_nested_dicts(self.cum_cms, lambda cm: rates2metrics(cm2rates(cm), 
                                                                                    self.exp_utils.class_names))
        return reports
    
    @staticmethod
    def is_rgb(year_str):  
        return int(year_str) >= 1998
    
    def _aggregate_year_cms(self, cms_dict):
        # years should be the lowest level of the dictionary
        keys_list = list(cms_dict.keys())
        if len(keys_list) == 0:
            return cms_dict
        elif isinstance(cms_dict[keys_list[0]], dict): # nested dic
            for key in cms_dict:
                cms_dict[key] = self._aggregate_year_cms(cms_dict[key])
            return cms_dict
        else:
            init_keys = list(cms_dict.keys())
            for key in init_keys:
                if 'overall' not in key:
                    cat = 'overall_rgb' if self.is_rgb(key) else 'overall_gray'
                    try:
                        cms_dict[cat] += cms_dict[key]
                    except KeyError:
                        cms_dict[cat] = cms_dict[key]
            cms_dict['overall'] = np.zeros_like(cms_dict[key])
            for cat in ['overall_gray', 'overall_rgb']:
                try:
                    cms_dict['overall'] += cms_dict[cat]
                except KeyError:
                    pass
            return cms_dict
            

    def _infer_sample(self, data, coords, dims, margins, seg_criterion = None):
        """Performs inference on one (multi-source) input accessed through dataset ds, with multiple outputs."""

        # compute dimensions of the output
        height, width = dims
        top_margin, left_margin, bottom_margin, right_margin = [int(m) for m in margins]

        # initialize accumulators
        output = torch.zeros((self.exp_utils.output_channels, height, width), dtype=torch.float32) 
        counts = torch.zeros((height, width), dtype=torch.float32)

        inputs, targets = data
        num_batches = len(inputs['input_main'])
        if self.evaluate:
            if seg_criterion is not None:
                seg_losses = np.zeros((num_batches,))
                valid_px_list = np.zeros((num_batches,))
        # iterate over batches of small patches
        for batch_idx in range(num_batches):
            # get the prediction for the batch
            input_data = self._dict_select_batch_and_todevice(inputs, batch_idx)
            if targets is not None:
                target_data = self._dict_select_batch_and_todevice(targets, batch_idx)
            with torch.no_grad():
                # forward pass
                t_main_actv, *_ = self.model(*input_data.values())
                # compute validation losses
                if self.evaluate:
                    if seg_criterion is not None:
                        seg_actv = t_main_actv
                        seg_target = target_data['target_tlm'] #.squeeze(1)
                        # main loss
                        try:
                            seg_mask = seg_target != seg_criterion.ignore_index
                            valid_px_list[batch_idx] = torch.sum(seg_mask).item()
                        except AttributeError:
                            valid_px_list[batch_idx] = seg_target.nelement()
                        seg_losses[batch_idx] = seg_criterion(seg_actv, seg_target).item()
                        
                        
                # move predictions to cpu
                main_pred = self.model.seg_normalization(t_main_actv).cpu()
            # accumulate the batch predictions
            for j in range(main_pred.shape[0]):
                x, y =  coords[batch_idx][j]
                padding = self.padding
                x_start, x_stop = x + padding, x + self.patch_size - padding
                y_start, y_stop = y + padding, y + self.patch_size - padding
                counts[x_start:x_stop, y_start:y_stop] += 1
                output[:, x_start:x_stop, y_start:y_stop] += main_pred[j, :, 
                                                                       padding:main_pred.shape[-2]-padding, 
                                                                       padding:main_pred.shape[-1]-padding]
                
        # normalize the accumulated predictions
        nopred_mask = counts==0
        counts = torch.unsqueeze(counts, dim = 0)
        mask = counts != 0

        rep_mask = mask.expand(output.shape[0], -1, -1)
        rep_counts = counts.expand(output.shape[0], -1, -1)
        output[rep_mask] = output[rep_mask] / rep_counts[rep_mask]
        
        # aggregate losses
        seg_loss, total_valid_px = None, None
        if self.evaluate:
            if seg_criterion is not None:
                seg_loss, total_valid_px = self._aggregate_batch_losses(seg_losses, 
                                                                        valid_px_list)

        # remove margins
        output = output[:, top_margin:height-bottom_margin, left_margin:width-right_margin]
        return output, (seg_loss, total_valid_px), nopred_mask
    
    @staticmethod            
    def _aggregate_batch_losses(loss_list, valid_px_list):
        total_valid_px = np.sum(valid_px_list, axis=0)
        try:
            seg_loss = np.average(loss_list, axis=0, weights=valid_px_list) 
        except ZeroDivisionError:
            seg_loss = 0
        return seg_loss, total_valid_px

    def infer(self, seg_criterion=None, *args, **kwargs):
        """
        Perform tile by tile inference on a dataset, evaluate and save outputs if needed

        Args:
            - criterion (nn.Module): criterion used for training, to be evaluated at validation as well to track 
                    overfitting
        """
        self.model.eval()
        
        if self.undersample > 1 or self.input_vrt_fn is None:
            # select sample to perform inference on
            df = self._select_samples()
            # create virtual mosaics (and set nodata values)
            self._get_vrt_from_df(df)
        # set the cumulative confusion matrix to 0
        if self.evaluate:
            self._reset_cm()       
            if seg_criterion is not None:
                seg_losses = [0] * len(df)
                valid_px_list = [0] * len(df)
                
        #create dataset
        ds = InferenceDataset(  self.input_fns,
                                self.input_vrt_fns, 
                                self.n_tiles,
                                exp_utils=self.exp_utils, 
                                batch_size=self.batch_size, 
                                patch_size=self.patch_size,
                                padding=self.padding,
                                target_fns=self.target_fns,
                                target_vrt_fns=self.target_vrt_fns,
                                input_nodata_val=self.input_vrt_nodata_val,
                                target_nodata_val=self.target_vrt_nodata_val)
        
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=None, # manual batching to obtain batches with patches from the same image
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn = lambda x : x,
            worker_init_fn=ds.seed_worker,
            generator=self.g,
        )
        
        if self.wandb_tracking:
            class_labels = {k: v for k, v in enumerate(self.exp_utils.class_names)}
            class_labels[self.exp_utils.i_out_nodata_val] = 'nodata'
            win_size = 256
            win_off = self.padding
            win_stop = win_off +  win_size
            # tile_track_list = ['2628_1122', '2661_1135', '2600_1129']
            tile_track_list = ['2628_1122']
        # iterate over dataset (tile by tile) 
        progress_bar = tqdm(zip(df.iterrows(), dataloader), total=len(df))
        for i, ((tile_idx, fns), (batch_data, target_tiles, coords, dims, margins, nodata_mask)) in enumerate(progress_bar):
            input_im_fn = fns.iloc[0]
            tile_num = self.exp_utils.tilenum_extractor(input_im_fn)
            progress_bar.set_postfix_str('Tile(s): {}'.format(tile_num))

            # compute forward pass and aggregate outputs
            outputs, losses, nopred_mask  = self._infer_sample(batch_data, coords, dims, margins, 
                                                  seg_criterion=seg_criterion)
            output = outputs
            seg_loss, valid_px = losses
            # store validation losses
            if self.evaluate:
                if seg_criterion is not None:
                    seg_losses[tile_idx] = seg_loss
                    valid_px_list[tile_idx] = valid_px

            # compute hard predictions and update confusion matrix
            output = output.numpy()
            
            # InferenceDataset returns nodata_mask as nparray
            nodata_mask = np.logical_or(nodata_mask, nopred_mask.numpy()) 
            # restore nodata values from inputs + missing predictions
            if np.any(nodata_mask):
                rep_mask = np.repeat(nodata_mask[np.newaxis, ...], output.shape[0], axis = 0) 
                output[rep_mask] = self.exp_utils.f_out_nodata_val
                
            output_hard = self._get_decisions(actv=output, 
                                                target_data=target_tiles, 
                                                nodata_mask=nodata_mask)
            
            # save in wandb table for vizualisation
            if self.wandb_tracking:
                if tile_num in tile_track_list:
                    with rasterio.open(input_im_fn, 'r') as f_input:
                        read_win_size = win_size / f_input.res[0]
                        win = Window(int(win_off / f_input.res[0]), 
                                     int(win_off / f_input.res[0]), 
                                     read_win_size, 
                                     read_win_size)
                        wandb_input_img = f_input.read( out_shape=( f_input.count,
                                                                        win_size,
                                                                        win_size),
                                                        resampling=Resampling.bilinear,
                                                        window=win)
                    target_tile = target_tiles['target_tlm'][0]
                    mask_img = wandb.Image(wandb_input_img.astype(np.uint8).transpose(1, 2, 0), 
                                           masks={"predictions": {
                                                        "mask_data": output_hard[win_off:win_stop, win_off:win_stop],
                                                        "class_labels": class_labels
                                                                },
                                                    "ground_truth": {
                                                        "mask_data": target_tile[win_off:win_stop, win_off:win_stop],
                                                        "class_labels": class_labels
                                                                  }
                    })
                    wandb_key = "pred_{}".format(tile_num)
                    wandb.log({wandb_key: mask_img})   
                       
                 
            # write outputs 
            if self.save_hard or self.save_soft:   
                writer = Writer(self.exp_utils, tile_num, input_im_fn)
                # main segmentation output
                writer.save_seg_result(self.output_dir, 
                                        save_hard = self.save_hard, output_hard = output_hard, 
                                        save_soft = self.save_soft, output_soft = output, 
                                        colormap = self.exp_utils.colormap)
        
        ###### compute metrics ######
        
        if self.evaluate:
            # compute confusion matrix and report
            reports = self._compute_metrics()
            # aggregate losses/errors/samples the validation set
            seg_loss = None if seg_criterion is None else np.average(seg_losses, axis = 0, 
                                                                                weights = valid_px_list)
            return self.cum_cms, reports, seg_loss
        else:
            return None
        
    def end(self):
        shutil.rmtree(self.tmp_dir)
        
    def __del__(self):
        try:
            shutil.rmtree(self.tmp_dir)
        except FileNotFoundError:
            pass
              
class TempInference(Inference):
    
    def __init__(self, model, 
                 file_list, 
                 exp_utils, 
                 output_dir = None, 
                 evaluate=True, 
                 save_hard=True, 
                 save_soft=True, 
                 save_temp_diff=False,
                 batch_size=32, 
                 patch_size=128, 
                 padding=4, 
                 num_workers=0, 
                 device=0, 
                 undersample=1,
                 random_seed=None, 
                 wandb_tracking=False,
                 wandb_log_pred=False,
                 fill_batch=True):
        super().__init__(model, 
                         file_list, 
                         exp_utils, 
                         output_dir, 
                         evaluate, 
                         save_hard, 
                         save_soft, 
                         batch_size, 
                         patch_size=patch_size, 
                         padding=padding, 
                         num_workers=num_workers, 
                         device=device, 
                         undersample=undersample, 
                         random_seed=random_seed, 
                         wandb_tracking=wandb_tracking)
        
        self.save_temp_diff = save_temp_diff
        self.wandb_log_pred = wandb_log_pred
        self.fill_batch = fill_batch
        
    def _check_col_names(self, df, source_names):
        """Get the column names used to read the dataset dataframe"""
        col_names = list(df)
        valid_names = []
        for name in source_names:
            if name in col_names: # look for a static (monotemporal) column
                valid_names.append(name)
            else:
                # look for a time series
                l = []
                t = 0
                temp_name = '{}_t{}'.format(name, t)
                found_input_col = False
                while temp_name in col_names:
                    l.append(temp_name)
                    found_input_col = True
                    t += 1
                    temp_name = '{}_t{}'.format(name, t)
                valid_names.append(l)
                if not found_input_col:
                    raise KeyError('Could not find {} column(s) in dataset csv file'.format(name))
        return valid_names
    
    def get_vrt_per_year(self, fns_per_position_df, data_name, nodata_val):
        """
        Year of acquisition should be 5th element of the filename, elements being separated with '_'.
        Assumes the nodata value is the same for all the acquisition year.
        """

        # remove years with no images
        fns_per_position_df = fns_per_position_df.loc[:,fns_per_position_df.notna().any(axis=0)] 
        year_df = fns_per_position_df.applymap(YEAR_EXTRACTOR)
        
        unique_years = np.unique(year_df.fillna(method='bfill').fillna(method='ffill'))
        keys = ['_'.join(data_name[0].split('_')[:-1] + [y]) for y in unique_years] 
        vrt_fns = {}
        new_nodata_val = None
        # create dataframe with acquisition years as columns which will contain filenames for each tile/row
        fns_per_year_df = pd.DataFrame(np.nan, index=fns_per_position_df.index, columns=keys)  
        for i, (y, key) in enumerate(zip(unique_years, keys)): # create a virtual mosaic for each acquisition year
            # get all the filenames for acquisition year 'y'
            valid_fns_df = fns_per_position_df[year_df == y]
            valid_fns_list = valid_fns_df.to_numpy(na_value='').flatten()
            valid_fns_list = valid_fns_list[valid_fns_list!='']
            # add a new column in fns_per_year_df corresponding to acquisition year 'y'
            new_col = valid_fns_df.dropna(axis=1, how='all').fillna(method='bfill', axis=1).iloc[:, 0]
            new_col[valid_fns_df.isna().all(axis=1)] = np.nan
            fns_per_year_df[key] = new_col
            # build the virtual mosaic
            vrt_fn, new_new_nodata_val = self.get_vrt(valid_fns_list, key, nodata_val)
            vrt_fns[key] = vrt_fn
            if i > 1:
                if new_new_nodata_val != new_nodata_val:
                    print('WARNING: found differing nodata values for different acquisition years. The nodata value of'
                          'the most recent year will be used for all years.')
            new_nodata_val = new_new_nodata_val         
                
        return vrt_fns, new_nodata_val, fns_per_year_df
        
    def _get_vrt_from_df(self, df): 
        """Build virtual mosaic rasters from files listed in dataframe df"""
        print('Building input virtual mosaics for inference...')
        self.input_fns = {}
        self.input_vrt_fns = {}
        self.input_vrt_nodata_val = {}
        self.input_fns_per_year_df = {}
        for input_name, item in zip(self.exp_utils.inputs, self.input_col_names):
            if isinstance(item, list):
                self.input_fns[input_name] = df[item]
                vrt_fns, new_nodata_val, temp_fns_per_year_df = self.get_vrt_per_year(
                                                                        self.input_fns[input_name], 
                                                                        item, 
                                                                        self.exp_utils.input_nodata_val[input_name])
                self.input_vrt_nodata_val[input_name] = new_nodata_val
                self.input_vrt_fns[input_name] = vrt_fns 
                self.input_fns_per_year_df[input_name] = temp_fns_per_year_df
            else:
                self.input_fns[input_name] = df[item]
                vrt_fn, new_nodata_val = self.get_vrt(self.input_fns[input_name], 
                                                      item, 
                                                      self.exp_utils.input_nodata_val[input_name])
                self.input_vrt_nodata_val[input_name] = new_nodata_val
                self.input_vrt_fns[input_name] = vrt_fn
            
        if self.evaluate or self.wandb_tracking:
            print('Building target virtual mosaics for inference...')  
            self.target_fns = {}           
            self.target_vrt_fns = {} 
            self.target_vrt_nodata_val = {}
            self.target_fns_per_year_df = {}
            for target_name, item in zip(self.exp_utils.targets, self.target_col_names):
                if isinstance(item, list):
                    self.target_fns[target_name] = df[item]
                    vrt_fns, new_nodata_val, temp_fns_per_year_df = self.get_vrt_per_year(
                                                                        self.target_fns[target_name], 
                                                                        item, 
                                                                        self.exp_utils.target_nodata_val[target_name])
                    self.target_vrt_nodata_val[target_name] = new_nodata_val
                    self.target_vrt_fns[target_name] = vrt_fns 
                    self.target_fns_per_year_df[target_name] = temp_fns_per_year_df
                else:
                    self.target_fns[target_name] = df[item]
                    vrt_fn, new_nodata_val = self.get_vrt(self.target_fns[target_name].dropna(), 
                                                          item, 
                                                          self.exp_utils.target_nodata_val[target_name])
                    self.target_vrt_nodata_val[target_name] = new_nodata_val
                    self.target_vrt_fns[target_name] = vrt_fn
        else:
            self.target_fns = None
            self.target_vrt_fns = None
            self.target_vrt_nodata_val = None
            self.target_fns_per_year_df = None
            
    @staticmethod            
    def _aggregate_batch_losses(loss_list, valid_px_list):
        """loss_list and valid_px_list have shape (n_batches,) or (n_batches, n_t)"""
        total_valid_px = np.sum(valid_px_list, axis=0)
        if np.any(valid_px_list):
            try:
                seg_loss = np.average(loss_list, axis=0, weights=valid_px_list) 
            except ZeroDivisionError: # no valid pixels for some of the time steps
                seg_loss = np.zeros_like(total_valid_px)
                for i in range(len(total_valid_px)):
                    if total_valid_px[i] != 0:
                        seg_loss[i] = np.average(loss_list[:, i], axis=0, weights=valid_px_list[:, i])
        else:
            if loss_list.ndim ==1:
                seg_loss = 0
            else:
                seg_loss = np.zeros(seg_loss.shape[1:])
        return seg_loss, total_valid_px
    
    @staticmethod
    def _weighted_dict_average(values, weights=None):
        """
        Performs a weighted average of a list of dictionaries
        'weights' should be a list of dictionaries with the same entries as the list of dictionaries 
        in 'values' 
        """
        aggregated_dict = defaultdict(partial(np.ndarray, (0, 2)))
        for batch_values, batch_weights in zip(values, weights):
            for year in batch_values:
                aggregated_dict[year] = np.concatenate(
                                        (aggregated_dict[year], np.array([[batch_values[year], batch_weights[year]]])), 
                                        axis=0)
        average = {}
        for year, arr in aggregated_dict.items():
            try:
                average[year] = np.average(arr[:,0], weights=arr[:,1])
            except ZeroDivisionError:
                average[year] = 0
        return average
    
    def _dict_average(self, values):
        aggregated_dict = defaultdict(partial(np.ndarray, (0,)))
        for batch_values in values:
            for year in batch_values:
                aggregated_dict[year] = np.append(aggregated_dict[year], batch_values[year])
        average = {}
        for year, arr in aggregated_dict.items():
            if np.all(np.isnan(arr)):
                average[year] = np.nan
            else:
                average[year] = np.nanmean(arr)
        return average
    
    def _infer_sample(self, 
                      data, 
                      coords, 
                      dims, 
                      margins, 
                      time_footprints, 
                      years,
                      seg_criterion=None, 
                      temp_criterion=None, 
                      temp_align_criterion=None):
        """Performs inference on one (multi-source) input accessed through dataset ds, with multiple outputs."""

        # compute dimensions of the output
        height, width = dims
        top_margin, left_margin, bottom_margin, right_margin = [int(m) for m in margins]
        
        inputs, targets = data
        n_t = time_footprints.shape[-1] #len(inputs['input_main'])
        num_batches = len(inputs['input_main'][0])
        # print('num_batches: {}'.format(num_batches))
        
        # initialize accumulators
        output = torch.zeros((self.exp_utils.output_channels, n_t, height, width),  # (n_classes, n_t, h, w)
                             dtype=torch.float32)
        counts = torch.zeros((n_t, height, width), dtype=torch.float32)

        if self.evaluate:
            if seg_criterion is not None:
                seg_losses = np.zeros((num_batches,))
                valid_px_list = np.zeros((num_batches,))
            if temp_criterion is not None:
                if self.fill_batch:
                    temp_losses_per_year = [None] * num_batches
                else:
                    temp_losses_per_year = np.zeros((num_batches,n_t-1)) 
                batch_temp_losses = np.zeros((num_batches,))
            if temp_align_criterion is not None:
                if self.fill_batch:
                    temp_align_losses_per_year = [None] * num_batches
                else:
                    temp_align_losses_per_year = np.zeros((num_batches,n_t-1)) 
                batch_temp_align_losses = np.zeros((num_batches,))
                
        years = years.to(self.device)
        # iterate over batches of small patches
        for batch_idx in range(num_batches):
            # get the prediction for the batch
            input_data = self._dict_select_batch_and_todevice(inputs, batch_idx)
            if self.fill_batch:
                temporal_mask = None
            else:
                temporal_mask = time_footprints[batch_idx].to(self.device)
            if targets is not None:
                target_data = self._dict_select_batch_and_todevice(targets, batch_idx) 
            with torch.no_grad():
                # forward pass
                t_main_actv = self.model(*input_data.values(), 
                                     years=years[batch_idx], 
                                     temporal_mask=temporal_mask) # (batch_size, n_classes, n_t, h, w)
                # compute validation losses
                if self.evaluate:
                    if seg_criterion is not None:
                        patch_idx_arr, t_idx_arr = torch.nonzero(years[batch_idx] == self.exp_utils.tlm_target_year, 
                                                                 as_tuple=True)
                        seg_actv = torch.zeros_like(t_main_actv[..., 0, :, :]) # (batch_size, C, H, W)
                        for (patch_idx, t_idx) in zip(patch_idx_arr, t_idx_arr): 
                            seg_actv[patch_idx] = t_main_actv[patch_idx, :, t_idx]
                        seg_target = target_data['target_tlm']
                        # main loss
                        try:
                            seg_mask = seg_target != seg_criterion.ignore_index
                            valid_px_list[batch_idx] = torch.sum(seg_mask).item()
                        except AttributeError:
                            valid_px_list[batch_idx] = seg_target.nelement()
                        seg_losses[batch_idx] = seg_criterion(seg_actv, seg_target).detach().cpu()
                        
                        if temp_criterion is not None:
                            n_t_this_batch = t_main_actv.shape[-3]
                            if n_t_this_batch > 1:
                                if self.fill_batch:
                                    (batch_temp_loss, batch_temp_loss_per_year), _ = \
                                        temp_criterion(t_main_actv, years=years[batch_idx], return_per_year=True)
                                else:
                                    raise NotImplementedError
                                batch_temp_losses[batch_idx] = batch_temp_loss.detach().cpu()
                                temp_losses_per_year[batch_idx] = batch_temp_loss_per_year
                        if temp_align_criterion is not None:
                            n_t_this_batch = t_main_actv.shape[-3]
                            if n_t_this_batch > 1:
                                if self.fill_batch:
                                    (batch_temp_align_loss, batch_temp_align_loss_per_year), _ = \
                                        temp_align_criterion(t_main_actv, years=years[batch_idx], return_per_year=True)
                                else:
                                    raise NotImplementedError
                                batch_temp_align_losses[batch_idx] = batch_temp_align_loss.detach().cpu()
                                temp_align_losses_per_year[batch_idx] = batch_temp_align_loss_per_year
                        
                        
                # move predictions to cpu
                main_pred = self.model.seg_normalization(t_main_actv).cpu() # (batch_size, n_classes, n_t, h, w)
            # accumulate the batch predictions
            for j in range(main_pred.shape[0]):
                # cropping coordinates to remove padding margins in the predicted patch 
                padding = self.padding
                xp_start, xp_stop = padding, main_pred.shape[-2]-padding
                yp_start, yp_stop = padding, main_pred.shape[-1]-padding
                # coordinates in the non-padded input image
                x, y =  coords[batch_idx][j]
                # bounding coordinates of predicted the patch (without padding margins) in the output accumulator
                x_start, x_stop = x + padding, x + self.patch_size - padding
                y_start, y_stop = y + padding, y + self.patch_size - padding
                if self.fill_batch:
                    # model predictions are stacked from the left
                    current_step = 0
                    for t in range(n_t):
                        if time_footprints[batch_idx][j][t]:
                            counts[t, x_start:x_stop, y_start:y_stop] += 1
                            output[:, t, x_start:x_stop, y_start:y_stop] += main_pred[j, :, current_step, 
                                                                                      xp_start:xp_stop, 
                                                                                      yp_start:yp_stop]
                            current_step += 1
                else:
                    counts[temporal_mask, x_start:x_stop, y_start:y_stop] += 1
                    output[:, temporal_mask, x_start:x_stop, y_start:y_stop] += main_pred[j, :, temporal_mask, 
                                                                                          xp_start:xp_stop, 
                                                                                          yp_start:yp_stop]
                
        # normalize the accumulated predictions
        nopred_mask = counts==0
        counts = counts.unsqueeze(0) 
        mask = counts != 0

        rep_mask = mask.expand(output.shape[0], -1, -1, -1)
        rep_counts = counts.expand(output.shape[0], -1, -1, -1)
        output[rep_mask] = output[rep_mask] / rep_counts[rep_mask]
        
        # aggregate losses in the batch
        seg_loss, total_valid_px = None, None
        temp_loss_per_year, temp_loss, temp_align_loss_per_year, temp_align_loss  = None, None, None, None
        if self.evaluate:
            if seg_criterion is not None:
                seg_loss, total_valid_px = self._aggregate_batch_losses(seg_losses, 
                                                                        valid_px_list)
            if temp_criterion is not None:
                temp_loss = np.mean(batch_temp_losses)
                temp_loss_per_year = self._dict_average(temp_losses_per_year)
            if temp_align_criterion is not None:
                temp_align_loss = np.mean(batch_temp_align_losses)
                temp_align_loss_per_year = self._dict_average(temp_align_losses_per_year)
                        
        # remove margins
        nopred_mask = nopred_mask[:, top_margin:height-bottom_margin, left_margin:width-right_margin]
        output = output[:, :, top_margin:height-bottom_margin, left_margin:width-right_margin]
        return output, \
                ((seg_loss, total_valid_px), (temp_loss_per_year, temp_loss), \
                    (temp_align_loss_per_year, temp_align_loss)), nopred_mask
    
    
    def infer(self, 
              seg_criterion=None, 
              temp_criterion=None, 
              temp_align_criterion=None,
              thresh=0.5):
        """
        Perform tile by tile inference on a dataset, evaluate and save outputs if needed

        Args:
            - criterion (nn.Module): criterion used for training, to be evaluated at validation as well to track 
                    overfitting
        """
        self.model.eval()
        
        try:
            reverse = self.model.reverse
        except AttributeError:
            reverse = False
        
        if self.undersample > 1 or self.input_vrt_fn is None:
            # select sample to perform inference on
            df = self._select_samples()
            # create virtual mosaics (and set nodata values)
            self._get_vrt_from_df(df)
        # set the cumulative confusion matrix to 0
        if self.evaluate:
            self._reset_cm()       
            if seg_criterion is not None:
                seg_losses = [0] * len(df)
                valid_px_list = [0] * len(df)
            if temp_criterion is not None:
                temp_losses = [None] * len(df)
                temp_losses_per_year = [None] * len(df)
            if temp_align_criterion is not None:
                temp_align_losses = [None] * len(df)
                temp_align_losses_per_year = [None] * len(df)
                          
        #create dataset
        ds = TempInferenceDataset(self.input_fns,
                                self.input_fns_per_year_df, 
                                self.input_vrt_fns, 
                                self.n_tiles,
                                exp_utils=self.exp_utils, 
                                batch_size=self.batch_size,
                                patch_size=self.patch_size,
                                padding=self.padding,
                                target_fns=self.target_fns,
                                target_vrt_fns=self.target_vrt_fns,
                                target_vrt_fns_per_year= self.target_fns_per_year_df,
                                input_nodata_val=self.input_vrt_nodata_val,
                                target_nodata_val=self.target_vrt_nodata_val,
                                fill_batch=self.fill_batch)

        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=None, # manual batching to obtain batches with patches from the same image
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=lambda x : x
        )
        
        if self.wandb_tracking:
            if self.wandb_log_pred:
                class_labels = {k: v for k, v in enumerate(self.exp_utils.class_names)}
                class_labels[self.exp_utils.i_out_nodata_val] = 'nodata'
                win_size = 256
                win_off = self.padding
                win_stop = win_off +  win_size
                stride = 4
                # tile_track_list = ['2628_1122', '2661_1135', '2600_1129']
                tile_track_list = ['2628_1122', '2661_1135']
            
        # iterate over dataset (tile by tile) 
        progress_bar = tqdm(zip(df.iterrows(), dataloader), total=len(df))
        wandb_table_data = [] 
        # wandb_table_data_soft = []
        # wandb_table_data_internals = []
        for (tile_idx, fns), \
            (batch_data, target_tiles, coords, time_footprints, years, dims, margins, nodata_mask, year_dic) \
            in progress_bar:
                
            input_im_fn = fns.loc['input_main_t0'] # will be used as a template filename
            tile_num = self.exp_utils.tilenum_extractor(input_im_fn)
                
            if batch_data is None:
                print('Skipping tile {}'.format(tile_num))
                with open(os.path.join(self.output_dir, 'skipped_tiles.txt'), 'a') as f:
                    f.write(tile_num + '\n')
                continue
            
            all_years = year_dic['input_main'] 
            
            progress_bar.set_postfix_str('Tile(s): {}'.format(tile_num))

            # compute forward pass and aggregate outputs
            # DEBUG
            output, losses, nopred_mask  = self._infer_sample(batch_data, 
                                                coords, 
                                                dims, 
                                                margins, 
                                                time_footprints,
                                                years=years,
                                                seg_criterion=seg_criterion, 
                                                temp_criterion=temp_criterion,
                                                temp_align_criterion=temp_align_criterion)
            # store validation losses
            if self.evaluate:
                (seg_loss, valid_px), (temp_loss_per_year, temp_loss), (temp_align_loss_per_year, temp_align_loss) = losses 
                if seg_criterion is not None:
                    seg_losses[tile_idx] = seg_loss
                    valid_px_list[tile_idx] = valid_px
                if temp_criterion is not None:
                    temp_losses[tile_idx] = temp_loss
                    if reverse:
                        temp_losses_per_year[tile_idx] = dict(zip(all_years[:-1], temp_loss_per_year))
                    else:
                        temp_losses_per_year[tile_idx] = dict(zip(all_years[1:], temp_loss_per_year))
                if temp_align_criterion is not None:
                    temp_align_losses[tile_idx] = temp_align_loss
                    if reverse:
                        temp_align_losses_per_year[tile_idx] = dict(zip(all_years[:-1], temp_align_loss_per_year))
                    else:
                        temp_align_losses_per_year[tile_idx] = dict(zip(all_years[1:], temp_align_loss_per_year))

            output = output.numpy()

            nodata_mask = np.logical_or(nodata_mask, nopred_mask.numpy())
            # restore nodata values found in the inputs or due to unused pixels
            if np.any(nodata_mask):
                rep_mask = np.repeat(nodata_mask[np.newaxis, :, :], output.shape[0], axis = 0)
                output[rep_mask] = self.exp_utils.f_out_nodata_val
            
            # choose which channels (i.e. time steps) are evaluated
            if target_tiles is None:
                eval_channel = None
            else:
                eval_channel = {}
                for key in target_tiles:
                    if key == 'target_multitemp': 
                        c = {}
                        for idx, y in enumerate(all_years):
                            if y in year_dic[key]:
                                c[y] = idx
                        eval_channel[key] = c
                    elif key == 'target_tlm':
                        eval_channel[key] = -1
                    else:
                        raise ValueError('eval_channel unknown for {}'.format(key))
                    
            # compute hard predictions and update confusion matrix
            output_hard = self._get_decisions(actv=output, 
                                                    target_data=target_tiles, 
                                                    nodata_mask=nodata_mask,
                                                    eval_channel=eval_channel,
                                                    thresh=thresh)    
                                                      
            if self.wandb_tracking:  
                if self.wandb_log_pred:
                    if tile_num in tile_track_list:
                        masked_image_list = [None] * len(all_years)
                        for i, y in enumerate(all_years):
                            with rasterio.open(fns.iloc[i], 'r') as f_input:
                                read_win_size = win_size / f_input.res[0]
                                win = Window(int(win_off / f_input.res[0]), int(win_off / f_input.res[0]), 
                                             read_win_size, 
                                             read_win_size)
                                wandb_input_img = f_input.read( out_shape=( f_input.count,
                                                                            win_size//stride,
                                                                            win_size//stride),
                                                                resampling=Resampling.nearest,
                                                                window=win
                                                                ).astype(np.uint8).transpose(1, 2, 0)
                            mask = {'predictions_{}'.format(y): { 
                                        "mask_data": output_hard[i][win_off:win_stop:stride, win_off:win_stop:stride],
                                        "class_labels": class_labels}
                                    }
                            if 'target_tlm' in target_tiles:
                                target_tile = target_tiles['target_tlm'][0]
                            else:
                                target_tile = None
                            if y == all_years[-1] and target_tile is not None:
                                mask['ground_truth'] = {    
                                                        "mask_data": target_tile[win_off:win_stop:stride, 
                                                                                 win_off:win_stop:stride],
                                                        "class_labels": class_labels
                                                        }
                            masked_image_list[i] = wandb.Image(wandb_input_img, 
                                                            masks=mask, 
                                                            caption=y)
                        wandb_table_data.append([tile_num ] + masked_image_list)
                        num_cols = max([len(row) for row in wandb_table_data])
                        columns = ['tile_num'] + ['pred_{}'.format(k) for k in range(num_cols - 1)]
                            
                        wandb_table_data = [row + [None]*(num_cols-len(row)) for row in wandb_table_data]
                        wandb_table_data = wandb.Table(columns=columns, data=wandb_table_data, allow_mixed_types=True)
                        wandb.log({"predictions": wandb_table_data})
                        print('Outputs logged to wandb')
                        

            # write outputs 
            if self.save_hard or self.save_soft: 
                
                # main segmentation output
                writer = Writer(self.exp_utils, tile_num, input_im_fn)
                for i, y in enumerate(all_years):
                    writer.save_seg_result(self.output_dir, 
                                            save_hard=self.save_hard, output_hard=output_hard[i], 
                                            save_soft=self.save_soft, output_soft=output[:, i], 
                                            suffix='_{}'.format(y),
                                            colormap=self.exp_utils.colormap)
                    if i > 0:
                        if self.save_temp_diff:
                            writer.save_seg_result(self.output_dir, 
                                                    save_hard=False, output_hard=None, 
                                                    save_soft=self.save_soft, 
                                                    output_soft=output[:, i] - output[:, i-1], 
                                                    suffix='_'.join(('_diff', all_years[i-1], y)), 
                                                    colormap=None)    
            # DEBUG      
                
        ###### compute metrics ######
        
        if self.evaluate:
            # compute confusion matrix and report
            reports = self._compute_metrics()
            # aggregate losses/errors/samples the validation set
            if seg_criterion is None:
                seg_loss = None  
            else: 
                try:
                    seg_loss = np.average(seg_losses, axis=0, weights=valid_px_list)
                except ZeroDivisionError:
                    seg_loss = 0
            temp_loss = None if temp_criterion is None else np.mean(temp_losses)
            temp_loss_per_year = None if temp_criterion is None else self._dict_average(temp_losses_per_year)
            temp_align_loss = None if temp_align_criterion is None else np.mean(temp_align_losses)
            temp_align_loss_per_year = None if temp_align_criterion is None else self._dict_average(temp_align_losses_per_year)
            return self.cum_cms, reports, \
                (seg_loss, temp_loss_per_year, temp_loss, temp_align_loss_per_year, temp_align_loss)
        else:
            return None