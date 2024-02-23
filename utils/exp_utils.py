import sys
import numpy as np
import torch
import os
from math import ceil
from torchvision.transforms import ColorJitter, GaussianBlur


project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

############## Constants and datasource-specific parameters ###################

# from https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
RGB_2_GRAY_WEIGHTS = torch.tensor([0.299, 0.587, 0.114]) 

# means and standard deviations
MEANS = {'SI2020':  torch.tensor([ 104.87825085, 114.34096987, 93.51435022]),
         'SItemp': np.load(os.path.join(project_dir, 'utils/SItemp100cm_1946_2020_means.npy'), 
                                allow_pickle=True).item(),
        'ALTI' : 1878.01851825}

STDS = {'SI2020' :  torch.tensor([50.58053258, 50.03784352, 47.48699987]),
        'SItemp': np.load(os.path.join(project_dir, 'utils/SItemp100cm_1946_2020_stds.npy'), 
                               allow_pickle=True).item(),
        'ALTI' : 1434.79671951}

# nodata value
I_NODATA_VAL = 255 #nodata value for integer arrays/rasters
F_NODATA_VAL = -1 #nodata value for float arrays/rasters

NODATA_VAL = {  'SI2020': None,
                'SItemp': 0,
                'TLM6c' : 255, 
                'ALTI' : -9999,
                'hard_predictions': I_NODATA_VAL,
                'soft_predictions': np.finfo(np.float32).max,
                'multitemp': 255}

# operators to use to check nodata       across bands, across pixels
NODATA_CHECK_OPERATOR = {   'SI2020': ['all', 'all'], 
                            'SItemp': ['all', 'any'],
                            'ALTI':   ['all', 'all'],
                            'TLM6c':               'all',
                            'multitemp':              'all'}

GET_OPERATOR = {'any': np.any, 'all': np.all}

# relative resolution of the datasources
RESOLUTION = {'SItemp': 1., 'ALTI': 1., 'TLM6c': 1., 'multitemp': 1.}

# number of channels
CHANNELS = {'SI2020': 3, 'SItemp': 3, 'ALTI': 1}

# class names
CLASS_NAMES = ['NF', 'F']

# number of classes
N_CLASSES = 2

# TLM translation for sub-tasks
nodata_mapping = np.full(251, fill_value = I_NODATA_VAL)
#                                                                   NF   OF  CF  SF  Gehoelzflaeche 
TARGET_CONVERSION_TABLE = {'TLM6c': np.concatenate((    np.array([  0,  1,  1,  0,  I_NODATA_VAL]), 
                                                        np.full(256-5, fill_value = I_NODATA_VAL))),
                           'multitemp': np.concatenate((np.array([  0,  1]), 
                                                        np.full(256-2, fill_value = I_NODATA_VAL)))}

TLM_TARGET_YEAR = 2020

# target colormap
COLORMAP = {  
            0: (0, 0, 0, 0),
            1: (255, 255, 255, 255),
            }

# class frequencies (used to weight the loss)
CLASS_FREQUENCIES_2020 = { # non-forest, forest (the latter including open forest and closed forest)
                            'all': {        'train': np.array([0.7530307998693597, 0.24696920013064028])},
                            'positives': {  'train': np.array([0.6334747563049722, 0.3665252436950278])}
                        }


CLASS_FREQUENCIES = { # non-forest, forest (the latter including open forest and closed forest) from TLM2022
                        'all': {        'train': np.array([0.7530307998693597, 0.24696920013064028])},
                        'positives': {  'train': np.array([0.6334747563049722, 0.3665252436950278])}
                    } 

# methods to extract the tile number from the filename
tilenum_extractor = lambda x, idx: os.path.splitext('_'.join(os.path.basename(x).split('_')[idx:idx+2]))[0]
TILENUM_EXTRACTOR = {   'SI2020': lambda x: tilenum_extractor(x, 2),
                        'SItemp': lambda x: tilenum_extractor(x, 2),
                        'ALTI': lambda x: tilenum_extractor(x, 3),
                        'TLM6c': lambda x: tilenum_extractor(x, 1),
                        'multitemp': lambda x: tilenum_extractor(x, 2)}

# isinstance(x, float) checks if x is nan
YEAR_EXTRACTOR = lambda x: x if isinstance(x, float) else os.path.splitext(x)[0].split('_')[4]
YEAR_INTERVAL_EXTRACTOR = lambda x : x if isinstance(x, float) else '_'.join(os.path.splitext(x)[0].split('_')[-2:])

IGNORE_INDEX = I_NODATA_VAL
IGNORE_FLOAT = F_NODATA_VAL #np.finfo(np.float32).max

MAX_YEAR = 2020
MIN_YEAR = 1946



############## ExpUtils class ##############################################

class ExpUtils:
    """
    Class used to define all parameters and methods specific to the data sources (inputs and targets) and the current
    experiment
    """

    def __init__(self, 
                 main_input_datasource, 
                 aux_input_datasource=None, 
                 multitemp_eval=True,
                 common_input_bands=None, 
                 augment_main_input=True,
                 jitter=0.1, 
                 sigma_max=0.3, 
                 grayscale_prob=0.5):
        """
        Args:
            - inputs_sources (list of str): input sources
            - target_datasource (str): dataset to extract targets from
            - common_input_bands (int): imposed number of bands for main inputs (mono or multi-temporal). Set to 1 to 
                                        convert all inputs to grayscale
        """

        # Get methods and parameters corresponding to input and target sources
        input_datasources = {'input_main': main_input_datasource}
        if aux_input_datasource is not None:
            input_datasources['input_aux'] = aux_input_datasource
            self.use_aux_input = True
        else:
            self.use_aux_input = False
        self.inputs = list(input_datasources.keys())
        self.n_input_sources = len(input_datasources)
        self.common_input_bands = common_input_bands
        self.input_channels = {
            'input_main': CHANNELS[main_input_datasource] if self.common_input_bands is None else self.common_input_bands
                                }
        if self.use_aux_input:
            self.input_channels['input_aux'] = CHANNELS[aux_input_datasource]
            
        tlm_target_source = 'TLM6c' # TLM with 6 classes 
        target_datasources = {'target_tlm': tlm_target_source}
        self.tlm_target_year = TLM_TARGET_YEAR
        if multitemp_eval is not None:
            val_multitemp_target_source = 'multitemp'
            target_datasources['target_multitemp'] = val_multitemp_target_source
        self.targets = list(target_datasources.keys())
        
        self.tilenum_extractor = TILENUM_EXTRACTOR[main_input_datasource]
        self.year_extractor = YEAR_EXTRACTOR
            
        self.input_means = {k: MEANS[v] for k, v in input_datasources.items()}
        self.input_stds = {k: STDS[v] for k, v in input_datasources.items()}
        
        if 'SItemp' in input_datasources.values():
            self.input_mean_ref = MEANS['SI2020'].clone().detach()
            self.input_std_ref = STDS['SI2020'].clone().detach()
            self.ref_year = '2020'
            
        if augment_main_input:
            color_jitter = ColorJitter(brightness=jitter, 
                                    contrast=jitter, 
                                    saturation=jitter, 
                                    hue=min(0.1, jitter))
            kernel_size = int(ceil(6*sigma_max)//2*2+1)
            gauss_blur = GaussianBlur(kernel_size=kernel_size, sigma=(1e-10, sigma_max))
            sim_gray = SimulateGrayscale(prob=grayscale_prob, 
                                         output_channels=self.input_channels['input_main'])
            self.augment_transforms = {'input_main': lambda x: sim_gray.simulate(color_jitter(gauss_blur(x)))}
        else:
            self.augment_transforms = {'input_main': None}
        if self.use_aux_input:
            self.augment_transforms['input_aux'] = None
                

        self.input_nodata_val = {k: NODATA_VAL[source] for k, source in input_datasources.items()}
        self.target_nodata_val = {k: NODATA_VAL[source] for k, source in target_datasources.items()}

        self.input_nodata_check_operator = {
            k: [GET_OPERATOR[op] for op in NODATA_CHECK_OPERATOR[source]] for k, source in input_datasources.items()
                                            } 
        
        self.target_nodata_check_operator = {
            k: GET_OPERATOR[NODATA_CHECK_OPERATOR[source]] for k, source in target_datasources.items()
                                            }
        
        self.target_conversion_table = {k: TARGET_CONVERSION_TABLE[source] for k, source in target_datasources.items()}
        
        # setup task(s)

        # main task
        self.n_classes = N_CLASSES
        self.colormap = COLORMAP
        self.class_names = CLASS_NAMES
        self.decision_func = self.binary_decision
        self.output_channels = 1 
        
        
        # nodata values for writing output rasters
        self.i_out_nodata_val = NODATA_VAL['hard_predictions']
        self.f_out_nodata_val = NODATA_VAL['soft_predictions']
        
        # nodata values for internal arrays/tensors (i.e. after pre-processing and before post-processing)
        self.i_nodata_val = I_NODATA_VAL
        self.f_nodata_val = F_NODATA_VAL

    ################# Methods for pre/post-processing #########################
    
    def preprocess_static_input(self, img, input_name, mode='eval'):
        """
        Number of channels should be the last dimension for the broadcasting to work.
        A nodata mask must be computed before this function, and use to avoid backpropagating loss on nodata pixels.
        """
        
        # convert into tensor 
        t_img = torch.from_numpy(img).float() 
        # scale using dataset-wise statistics
        t_img = torch.movedim((t_img - self.input_means[input_name]) / self.input_stds[input_name], (2, 0, 1), (0, 1, 2))
        
        augment = self.augment_transforms[input_name] is not None and mode=='train'
        if augment:
            # rescale pixel values to [0, 1] (requires by some data augmentation functions)
            img_min_val = torch.min(t_img)# np.min(arr_img)
            img_max_val = torch.max(t_img)# np.max(arr_img)
            try:
                t_img = ((t_img - img_min_val) / (img_max_val - img_min_val))
            except RuntimeWarning as e:
                if img_min_val == img_max_val:
                    augment = None # no need to do data augmentation on a constant image
                else:
                    raise RuntimeWarning(e)
        
            # write image for vizualisation 
            # data = (t_img * 255).numpy().astype(np.uint8) #.transpose(2, 0, 1)
            # with rasterio.open('patch.png', 'w', driver='PNG',width=256, height=256, count=data.shape[0], dtype='uint8') as f:
            #     f.write(data)
        
        # apply augmentation
        #if augment:
            t_img = self.augment_transforms[input_name](t_img) #ColorJitter requires non-negative values
            
            # write image for vizualisation
            # new_data = t_img.numpy()
            # new_data = (new_data * 255).astype(np.uint8)
            # with rasterio.open('patch_aug.png', 'w', driver='PNG',width=256, height=256, count=new_data.shape[0], dtype='uint8') as f:
            #     f.write(new_data)
                
            # scale values back to original scale
            t_img = t_img * (img_max_val - img_min_val) + img_min_val
        # scale using dataset-wise statistics
        if self.input_channels[input_name] < t_img.shape[0]:
            if self.input_channels[input_name]==1:
                t_img = torch.sum((RGB_2_GRAY_WEIGHTS * t_img.movedim((1, 2, 0),(0, 1, 2))).movedim((2, 0, 1), (0, 1, 2)), 
                                  axis=0, 
                                  keepdim=True)
            else:
                raise RuntimeError('Cannot reduce {} bands into {} bands'.format(t_img.shape[0], 
                                                                                 self.input_channels[input_name]))
        elif self.input_channels[input_name] > t_img.shape[0]:
            raise NotImplementedError 
        return t_img
        
        
    def preprocess_multitemp_input(self, img_list, input_name, year_list=None): 
        """
        Uses acquisition year-specific, or global statistics to normalize the data.
        Number of channels should be the last dimension for the broadcasting to work.
        A nodata mask must be computed before this function, and use to avoid backpropagating loss on nodata pixels.
        """
        if year_list is None:
            raise RuntimeError('Argument "year_list" is necessary to normalize data per acquisition year')
        else:
            for i, (year, data) in enumerate(zip(year_list,img_list)):
                data = torch.from_numpy(data).float()
                try: # year for which statistics are available on a subset of the dataset only
                    mean_year_loc, mean_ref_loc = self.input_means[input_name][year]
                    std_year_loc, std_ref_loc = self.input_stds[input_name][year]
                    # this always gives as many bands as in the ref year
                    data = ((data - mean_year_loc) / std_year_loc * std_ref_loc + mean_ref_loc - self.input_mean_ref) \
                            / self.input_std_ref
                except ValueError: # year for which the dataset-wide statistics are available
                    mean_year = self.input_means[input_name][year]
                    std_year = self.input_stds[input_name][year]
                    data = (data - mean_year) / std_year
                except KeyError:
                    if year == self.ref_year:
                        data = (data - self.input_mean_ref) / self.input_std_ref
                    else:
                        # find closest acquisition year
                        year_diff = [abs(int(y) - int(year)) for y in self.input_means[input_name].keys()]
                        idx_closest_year = np.argmin(year_diff)
                        closest_year = list(self.input_means[input_name].keys())[idx_closest_year]
                        
                        mean_year_loc, mean_ref_loc = self.input_means[input_name][closest_year]
                        std_year_loc, std_ref_loc = self.input_stds[input_name][closest_year]
                        # this always gives as many bands as in the ref year
                        data = ((data - mean_year_loc) / std_year_loc * std_ref_loc + mean_ref_loc - self.input_mean_ref) \
                                / self.input_std_ref
                        print('Warning: Data mean and std not found for year {}, using year {} instead'.format(year, closest_year))
                if self.common_input_bands is not None:
                    if self.common_input_bands == data.shape[-1]:
                        continue
                    elif self.common_input_bands == 1:
                        data = torch.mean(data, dim=-1, keepdim=True)
                    else:
                        raise NotImplementedError
                

                img_list[i] = torch.movedim(data.float(), (2, 0, 1), (0, 1, 2))

        return img_list

    def preprocess_static_inputs(self, img_dict, mode='eval'):
        return {input_name: self.preprocess_static_input(img, 
                                                         input_name, 
                                                         mode=mode) for input_name, img in img_dict.items()}
        
    def preprocess_mixed_inputs(self, img_dict, year_dict=None):
        processed_inputs = {}
        for key, data in img_dict.items():
            if isinstance(data, list):
                try:
                    year_list_i = year_dict[key]
                except KeyError:
                    year_list_i = None
                processed_inputs[key] = self.preprocess_multitemp_input(data, key, year_list_i)
            else:
                processed_inputs[key] = self.preprocess_static_input(data, key)
        return processed_inputs
    
    
    def preprocess_year_info(self, str_year_list):
        year_list = []
        for str_y in str_year_list:
            year_list.append(float(str_y))
        return torch.tensor(year_list) 
    
    def preprocess_training_target(self, x, name):
        conversion_table = self.target_conversion_table[name]
        if conversion_table is not None:
            x = conversion_table[x]
        return torch.from_numpy(x).squeeze(-1)
        
    def preprocess_inference_target(self, x, name):
        conversion_table = self.target_conversion_table[name]
        if conversion_table is not None:
            x = conversion_table[x]
        return np.squeeze(x, axis=-1)

    def postprocess_target(self, targets):
        return targets
    
    

    ######################## Methods to check nodata ##########################

    def static_inputs_nodata_check(self, input_dic):
        """
        each input should have 3 dimensions (height, width, bands)
        """
        check = False
        for key, val in input_dic.items():
            op1, op2 = self.input_nodata_check_operator[key]
            if self.input_nodata_val[key] is None:
                check = check or False
            else:
                data = np.reshape(val, (-1, val.shape[-1])) # flatten along height and width
                check = check or op2(op1(data == self.input_nodata_val[key], axis = -1), axis = 0)
        return check

    def mixed_inputs_nodata_check(self, input_dic): # static and temporal inputs
        check = False
        for key, val in input_dic.items():
            op1, op2 = self.input_nodata_check_operator[key]
            if self.input_nodata_val[key] is None:
                check = check or False
            else:
                if isinstance(val, list):
                    for d in val:
                        data = np.reshape(d, (-1, d.shape[-1])) # flatten along height and width
                        check = check or op2(op1(data == self.input_nodata_val[key], axis = -1), axis = 0)
                else:
                    data = np.reshape(val, (-1, val.shape[-1])) 
                    check = check or op2(op1(data == self.input_nodata_val[key], axis = -1), axis = 0)
        return check
    
    @staticmethod
    def single_band_nodata_check(data, nodata_val, nodata_check_operator):
        """
        data should have 2 dimensions (height, width)
        """
        if nodata_val is None:
            return False
        else:
            check = nodata_check_operator(np.ravel(data) == nodata_val)
        return check

    def target_nodata_check(self, data):
        return self.single_band_nodata_check(data, self.target_nodata_val['target_tlm'], 
                                             self.target_nodata_check_operator['target_tlm'])


    ######## Methods for converting soft predictions to hard predictions ######

    def binary_decision(self, output, thresh=0.5): 
        if isinstance(output, torch.Tensor):
            output_hard = (output > thresh)
        else: # output is a numpy array
            output_hard = (output > thresh).astype(np.uint8)
        return output_hard    
class SimulateGrayscale:
    def __init__(self, prob=0.5, output_channels=3, std_noise=0.1):
        self.output_channels = output_channels
        self.std_noise = torch.ones(3) * std_noise
        if self.output_channels == 1:
            self.prob = 1
        else:
            self.prob = prob
        
    def simulate(self, x):
        if np.random.random() <= self.prob: 
            valid = False
            while not valid:
                weights = RGB_2_GRAY_WEIGHTS + torch.normal(0, self.std_noise) 
                weights = weights / torch.sum(weights)   
                valid = torch.all(weights > 0).item()
            new_x = torch.sum(x.T * weights, axis=-1).unsqueeze(0)
            new_x = new_x.repeat(self.output_channels, 1, 1)
            return new_x
        else:
            return x 
            

    




  
        