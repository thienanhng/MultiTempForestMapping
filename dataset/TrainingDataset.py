import os
import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset
import rasterio
from rasterio.errors import RasterioError, RasterioIOError
from rasterio.enums import Resampling
from math import ceil
import pandas as pd
import random
import torchvision.transforms.functional as F

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'Data')

class TrainingDataset(IterableDataset):
    """
    Dataset for training. Generates random small patches over the whole training set.
    """

    def __init__(self, 
                 dataset_csv, 
                 n_input_sources, 
                 exp_utils, 
                 control_training_set=True, 
                 patch_size=128,
                 num_patches_per_tile=32, 
                 verbose=False, 
                 augment_flip=True,
                 undersample=1,
                 **kwargs):
        """
        Args:
            - exp_utils (ExpUtils)
            - n_neg_samples (int): number of negative samples (i.e. containing class 0 only) to use
        """

        self.n_input_sources = n_input_sources
        self.control_training_set = control_training_set
        self.augment_flip = augment_flip
        self.patch_size = patch_size
        self.num_patches_per_tile = num_patches_per_tile
        self.exp_utils = exp_utils
        self.verbose = verbose
        self.undersample = undersample
        self.fns = pd.read_csv(dataset_csv)
        self._check_df_columns()
        
        self.n_fns_all = len(self.fns)

        # store filenames of positive and negative examples separately
        if control_training_set:
            self._split_fns()
            self.n_positives = len(self.fns_positives)
            self.n_negatives = len(self.fns_negatives)
        else:
            self.fns_positives = None
            self.fns_negatives = None
            self.n_positives = None
            self.n_negatives = None
            
    def _check_df_columns(self):
        col_names = list(self.fns)
        # target
        if self.exp_utils.targets[0] in col_names:
            self.target_col_name = self.exp_utils.targets[0]
        else:
            raise KeyError('{} column not found in the dataset csv file'.format(self.exp_utils.targets[0]))
        # input(s)
        self._check_df_input_columns()
        # counts            
        if self.control_training_set:
            for i in range(1, self.exp_utils.n_classes):
                if 'count_{}'.format(i) not in col_names:
                    raise KeyError('Could not find count_{} column(s) in dataset csv file'.format(i))
                
    def _check_df_input_columns(self):
        col_names = list(self.fns)
        self.input_col_names = []
        for name in self.exp_utils.inputs:
            if name in col_names:
                self.input_col_names.append(name)
            else:
                raise KeyError('Could not find {} column in dataset csv file'.format(name))

    def _split_fns(self):
        """
        Creates two dataframes self.fns_positives and self.fns_negatives which store positive and negative filenames 
        separately
        """
        positive_counts = self.fns.loc[:, ['count_' + str(i) for i in range(1, self.exp_utils.n_classes)]]   
        positives_mask = positive_counts.any(axis=1).to_numpy()
        self.fns_positives = self.fns[positives_mask]
        self.fns_negatives = self.fns[~positives_mask]
        

    def select_tiles(self, n_neg_samples):
        """
        Fills self.fn with a given number of negative samples
        """
        n_neg_samples = min(n_neg_samples, self.n_negatives)
        if self.undersample > 1:  
            # select positives samples such that the total number of samples (positives+negatives) corresponds to the 
            # undersampling ratio 
            n_tiles = min(self.n_fns_all // self.undersample - n_neg_samples, self.n_positives)
            idx = random.sample(range(self.n_positives), n_tiles)   
            fns_pos = self.fns_positives.iloc[idx]
        else:
            fns_pos = self.fns_positives
        if n_neg_samples == 0: # do not use negative samples
            self.fns = fns_pos
        elif n_neg_samples < self.n_negatives: # pick negative samples randomly
            draw_idx = np.random.choice(self.n_negatives, size=(n_neg_samples,), replace = False)
            self.fns = pd.concat([fns_pos, self.fns_negatives.iloc[draw_idx]], ignore_index=True)
        else: # use all negative samples
            self.fns = pd.concat([fns_pos, self.fns_negatives], ignore_index=True)
        print('Using {} training samples out of {}'.format(len(self.fns), self.n_fns_all))
            
    def shuffle(self):
        # shuffle samples
        self.fns = self.fns.sample(frac=1).reset_index(drop=True)
        
    @staticmethod
    def seed_worker(worker_id):
        """from https://pytorch.org/docs/stable/notes/randomness.html"""
        worker_seed = torch.initial_seed() % 2**32
        # print('Worker seed {}: {}'.format(worker_id, worker_seed))
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _get_worker_range(self, fns):
        """Get the range of tiles to be assigned to the current worker"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # WARNING: when several workers are created they all have the same numpy random seed but different torch random 
        # seeds. 
        seed = torch.randint(low=0,high=2**32-1,size=(1,)).item()
        np.random.seed(seed) # set a different seed for each worker

        # define the range of files that will be processed by the current worker: each worker receives 
        # ceil(num_filenames / num_workers) filenames
        num_files_per_worker = ceil(len(fns) / num_workers)
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(len(fns), (worker_id+1) * num_files_per_worker)

        return lower_idx, upper_idx

    def _stream_tile_fns(self, lower_idx, upper_idx):
        """Generator providing input and target paths tile by tile from lower_idx to upper_idx"""
        for idx in range(lower_idx, upper_idx):
            yield self.fns.iloc[idx]

    def _extract_multi_patch(self, data, scales, x, xstop, y, ystop):
        """
        Extract a patch from multisource data given the relative scales of the sources and boundary coordinates
        """
        return {k: self._extract_patch(v,scales[k], x, xstop, y, ystop) for k, v in data.items()}

    def _extract_patch(self, data, x, xstop, y, ystop):
        return data[y:ystop, x:xstop]

    def _generate_patch(self, data, num_skipped_patches, coord = None):
        """
        Generates a patch from the input(s) and the targets, randomly or using top left coordinates "coord"

        Args:
            - data (list of (list of) tensors): input and target data
            - num_skipped_patches (int): current number of skipped patches (will be updated)
            - coord: top left coordinates of the patch to extract, in the coarsest modality

        Output:
            - patches (list of (list of) tensors): input and target patches
            - num_skipped_patches (int)
            - exit code (int): 0 if success, 1 if IndexError or invalid patch (due to nodata)
            - (x,y) (tuple of ints): top left coordinates of the extracted patch
        """

        input_data, target_data, *metadata = data
        # find the coarsest data source
        height, width = target_data.shape[:2] 

        if coord is None: # pick the top left pixel of the patch randomly
            x = np.random.randint(0, width-self.patch_size)
            y = np.random.randint(0, height-self.patch_size)
            # print(x)
        else: # use the provided coordinates
            x, y = coord
            
        # extract the patch
        try:
            xstop = x + self.patch_size
            ystop = y + self.patch_size
            # extract input patch
            input_patch = self._extract_multi_patch(input_data, x, xstop, y, ystop)
            # extract target patch
            target_patch = self._extract_patch(target_data, x, xstop, y, ystop)
        except IndexError:
            if self.verbose:
                print("Couldn't extract patch (IndexError)")
            return (None, num_skipped_patches, 1, (x, y))

        # check for no data
        skip_patch = self.exp_utils.target_nodata_check(target_patch) or self._inputs_nodata_check(input_patch) 
        if skip_patch: # the current patch is invalid
            num_skipped_patches += 1
            return (None, num_skipped_patches, 1, (x, y))

        # preprocessing (needs to be done after checking nodata)
        if self.augment_flip:
            hflip, vflip = torch.bernoulli(torch.full((2,), 0.5)) # whether to flip the data horizontally and vertically
        else:
            hflip, vflip = False, False
        input_patch = self._preprocess_inputs(input_patch, hflip, vflip, *metadata)
        target_patch = self._preprocess_targets(target_patch, hflip, vflip)
        patches = [input_patch, target_patch]
        
        return (patches, num_skipped_patches, 0, (x, y))
    
    def _flip(self, patch, hflip=False, vflip=False):
        # horizontal and vertical flip
        if hflip:
            patch = F.hflip(patch)
        if vflip:
            patch = F.vflip(patch)
        return patch
    
    def _preprocess_inputs(self, patches, hflip=False, vflip=False, *args, **kwargs):
        """assumes "patches" variable contains [input_image, DEM]"""
        # preprocessing (including non-geometric augmentation)
        patches = list(self.exp_utils.preprocess_static_inputs(patches, mode='train').values())
        # geometric augmentation
        patches = [self._flip(p, hflip, vflip) for p in patches]
        return patches
    
    def _preprocess_targets(self, target_patch, hflip=False, vflip=False):
        target_patch = self.exp_utils.preprocess_training_target(target_patch, 'target_tlm')
        target_patch = self._flip(target_patch, hflip, vflip)
        return target_patch
            
    def _inputs_nodata_check(self,patches):
        return self.exp_utils.static_inputs_nodata_check(patches) 
    
    def _harmonized_reader(self, fp, res=1.0):
        """Automatically resamples to 1m resolution"""
        rescale_factor =  fp.res[0] / res # assumes the pixels are square (horizontal resolution = vertical resolution)
        data = fp.read(out_shape=(fp.count,
                                  int(fp.height * rescale_factor),
                                  int(fp.width * rescale_factor)),
                       resampling=Resampling.bilinear)    
        return data

    def _read_tile(self, df_row):
        """
        Reads the files in files img_fn and target_fn
        Args:
            - img_fn (tuple of str): paths to the inputs
            - target_fn (str): path to the target file
        Output:
            - img_data (dict of tensors)
            - target_data (tensor)
        """
        try: # open files
            img_fp = {k: rasterio.open(os.path.join(DATA_DIR, fn), "r") for k, fn in zip(self.input_col_names, 
                                                                 list(df_row[self.input_col_names]))}
            target_fp = rasterio.open(os.path.join(DATA_DIR, df_row[self.target_col_name]), "r")
        except (RasterioIOError, rasterio.errors.CRSError) as e:
            print('WARNING: {}'.format(e))
            return None

        # read data for each input source and for the targets
        try:
            img_data = {}
            for key, fp in img_fp.items():  
                img_data[key] = np.moveaxis(self._harmonized_reader(fp), 
                                            (1, 2, 0), (0, 1, 2))
            target_data = target_fp.read(1)
        except RasterioError as e:
            print("WARNING: Error reading file, skipping to the next file")
            return None

        # close file pointers and return data
        for fp in img_fp.values():
            fp.close()
        target_fp.close()

        return img_data, target_data

    def _get_patches_from_tile(self, fns): 
        """Generator returning patches from one tile"""
        num_skipped_patches = 0
        # read data
        data = self._read_tile(fns)
        if data is None:
            return #skip tile if couldn't read it

        # yield patches one by one
        n_generated_patches = 0
        n_trials = 0
        
        while n_generated_patches < self.num_patches_per_tile and n_trials < 2 * self.num_patches_per_tile:
            n_trials += 1
            data_patch, num_skipped_patches, code, _ = self._generate_patch(data, num_skipped_patches, None)
            if code != 1:
                n_generated_patches += 1
                yield data_patch

        if num_skipped_patches>0 and self.verbose:
            print("We skipped %d patches on %s" % (num_skipped_patches, fns[0]))

    def _stream_patches(self):
        """Generator returning patches from the samples the worker calling this function is assigned to"""
        lower_idx, upper_idx = self._get_worker_range(self.fns)
        for fns in self._stream_tile_fns(lower_idx, upper_idx): #iterate over tiles assigned to the worker
            yield from self._get_patches_from_tile(fns) #generator 

    def __iter__(self):
        if self.verbose:
            print("Creating a new TrainingDataset iterator")
        return iter(self._stream_patches())
   
    
    
class TempTrainingDataset(TrainingDataset):
    
    def __init__(self, dataset_csv, n_input_sources, exp_utils, control_training_set = True, verbose=False, **kwargs):
        
        super().__init__(dataset_csv, n_input_sources, exp_utils, control_training_set, verbose=verbose, **kwargs)
                
    def _check_df_input_columns(self):
        col_names = list(self.fns)

        self.input_col_names = []
        for name in self.exp_utils.inputs:
            if name in col_names: # look for a static (monotemporal) column
                self.input_col_names.append(name)
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
                self.input_col_names.append(l)
                if not found_input_col:
                    raise KeyError('Could not find {} column(s) in dataset csv file'.format(name))

            
    def _read_tile(self, df_row):
        """
        Reads the files in files img_fn and target_fn
        Args:
            - img_fn (tuple of str): paths to the inputs
            - target_fn (str): path to the target file
        Output:
            - img_data (list of tensors)
            - target_data (tensor)
        """
        try: # open files 
            img_fp = {}
            img_years = {}
            for item in self.input_col_names:
                if isinstance(item, list): # time series
                    key = '_'.join(item[0].split('_')[:-1])
                    fp = [None] * len(item)
                    years = [None] * len(item)
                    for i, name in enumerate(item):
                        try:
                            fn = df_row[name]
                            fp[i] = rasterio.open(os.path.join(DATA_DIR, fn), "r") 
                            years[i] = self.exp_utils.year_extractor(fn)
                        except TypeError: # filename is empty (reached end of the time series)
                            if i == 0: # no input time series to read
                                return None 
                            else:
                                break
                    img_fp[key] = fp
                    img_years[key] = years
                else: # single image
                    img_fp[item] = rasterio.open(os.path.join(DATA_DIR, df_row[item]), "r")
                    img_years[item]= None
            target_fp = rasterio.open(os.path.join(DATA_DIR, df_row[self.target_col_name]), "r")
        except (RasterioIOError, rasterio.errors.CRSError) as e:
            print('WARNING: {}'.format(e))
            return None
        
        # read data for each input source and for the targets
        try:
            img_data = {}
            for key, fp in img_fp.items():
                if isinstance(fp, list): # time series
                    img_data[key] = [np.moveaxis(self._harmonized_reader(fp_i), 
                                               (1, 2, 0), 
                                               (0, 1, 2)) 
                                                                                    for fp_i in fp if fp_i is not None]
                    img_years[key] = [year for year in img_years[key] if year is not None]
                else:  #single image
                    img_data[key] = np.moveaxis(self._harmonized_reader(fp), 
                                              (1, 2, 0), 
                                              (0, 1, 2))
            target_data = target_fp.read(1)
        except RasterioError as e:
            print("WARNING: Error reading file, skipping to the next file")
            return None

        # close file pointers and return data
        for fp in img_fp.values():
            if isinstance(fp, list):
                for fp_i in fp:
                    if fp_i is not None:
                        fp_i.close()
            else:
                fp.close()
        target_fp.close()

        return img_data, target_data, img_years
    
    def _extract_multi_patch(self, data, x, xstop, y, ystop):
        """
        Extract a patch from multisource data given the relative scales of the sources and boundary coordinates
        """
        patches = {}
        for key, val in data.items(): 
            if isinstance(val, list):
                patches[key] = [self._extract_patch(val[j], x, xstop, y, ystop) for j in range(len(val))]
            else:
                patches[key] = self._extract_patch(val, x, xstop, y, ystop)
        return patches
                
    def _preprocess_inputs(self, patches, hflip=False, vflip=False, years=None): 
        """ assumes the main input is multi-temporal and the auxiliary input is mono-temporal """
        # preprocess image inputs and targets
        temp_patches, static_patch = self.exp_utils.preprocess_mixed_inputs(patches, years).values()
        # preprocess metadata (date information)
        years = self.exp_utils.preprocess_year_info(years['input_main'])
        static_patch = self._flip(static_patch, hflip, vflip)
        temp_patches = [self._flip(p, hflip, vflip) for p in temp_patches]

        return temp_patches, static_patch, years.unsqueeze(0) # unsqueeze to collate
    
    def _inputs_nodata_check(self,patches): 
        return self.exp_utils.mixed_inputs_nodata_check(patches)
    
    def _generate_patch(self, data, num_skipped_patches, coord = None):
        """Generates input and target data as well as metadata for one patch"""
        patch, *other_outputs = super()._generate_patch(data, num_skipped_patches, coord)
        return (patch, *other_outputs)