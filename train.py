import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import TrainingDataset, TempTrainingDataset, collate_variable_length_series
from models import Unet, RecurrentUnet, GRUUnet, NonRecurrentUnet
import utils
from utils import ExpUtils
from utils.train_utils import MyBCELoss, MyBCEWithLogitsLoss, MyTemporalMSELoss, MyGradDotTemporalLoss, \
                                MyTemporalCELoss, MyGradNormTemporalLoss
from copy import deepcopy
from numpy.core.numeric import NaN
import random

def train(output_dir, 
            main_input_source,
            aux_input_source,
            train_csv_fn,
            val_csv_fn,
            temp, 
            num_epochs=30,
            random_seed=0,
            new_history=True,
            starting_model_fn=None,
            resume_training=False, 
            freeze_matching_params=0,
            skip_validation=False,
            validation_period=1,
            model_arch='GRUUnet',
            undersample_training=1,
            undersample_validation=1,
            num_patches_per_tile=8,
            n_negative_samples=[20],
            negative_sampling_schedule=[30],
            batch_size=8,
            patch_size=128,
            lr_fe=1e-5,
            lr_temp=1e-3,
            update_period=1,
            bn_momentum=0.1,
            augment_flip=True,
            augment_vals=True,
            gauss_blur_sigma=0.5,
            color_jitter=0.5,
            grayscale_prob=0.5,
            std_gray_noise=0.1,
            lambda_temp=0.01,
            temp_loss='MSE',
            lambda_temp_align=0,
            temp_align_loss=None,
            scale_by_norm=True, # for ablation study
            asym_align=False, # for ablation study
            weight_temp_loss=True,
            reverse=False,
            gru_irreg=True,
            gru_kernel_size=7,
            gru_last_actv='sigmoid',
            gru_init='last',
            gru_input='df',
            gru_norm_dt=False,
            common_input_bands=None, 
            normalize_temp_inputs='per_year',
            num_workers_train=8,
            num_workers_val=4,
            debug=False,
            no_user_input=True):
    
    args_dict = locals().copy()
        
    torch.autograd.set_detect_anomaly(debug)

    exp_name = os.path.basename(output_dir)
    log_fn = os.path.join(output_dir, 'training','{}_metrics.pt'.format(exp_name))
    model_fn = os.path.join(output_dir, 'training', '{}_model.pt'.format(exp_name))

    ############ Check paths ###########
    if resume_training:
        if starting_model_fn is not None:
            if os.path.isfile(starting_model_fn):
                if model_fn == starting_model_fn:
                    raise RuntimeError('Current model file and starting point model file are identical ({}), aborting '
                                       'to avoid overwriting the starting point model'.format(model_fn))
                print('Training with starting point {}'.format(starting_model_fn))
            else:
                raise FileNotFoundError('Could not find starting point {}'.format(starting_model_fn))
    if os.path.isfile(model_fn):
        if os.path.isfile(log_fn):
            if resume_training:
                print('Resuming the training process, {} and {} will be updated.'.format(log_fn, model_fn))
            else:
                print('WARNING: Training from scratch, {} and {} will be overwritten'.format(log_fn, model_fn))
                if not no_user_input:
                    print('Continue? (yes/no)')
                    while True:
                        proceed = input()
                        if proceed == 'yes':
                            break
                        elif proceed == 'no':
                            return
                        else:
                            print('Please answer by yes or no')
                            continue
        else:
            if resume_training and not(new_history):
                raise FileNotFoundError('Cannot resume training, {} does not exist'.format(log_fn))
            elif not os.path.isdir(os.path.dirname(log_fn)):
                print('Directory {} does not exist, it will be created'.format(os.path.dirname(log_fn)))
                os.makedirs(os.path.dirname(log_fn))
    else:
        if resume_training and starting_model_fn is None:
            raise FileNotFoundError('Cannot resume training, {} does not exist and "starting_model_fn" was not used'\
                                        .format(model_fn))
        elif not os.path.isdir(os.path.dirname(model_fn)):
            print('Directory {} does not exist, it will be created'.format(os.path.dirname(model_fn)))
            os.makedirs(os.path.dirname(model_fn))
            
    if not os.path.isfile(train_csv_fn):
        raise FileNotFoundError('Could not find specified file {}'.format(train_csv_fn))
    if val_csv_fn is None:
        skip_validation = True
    else:
        if not os.path.isfile(val_csv_fn):
            raise FileNotFoundError('Could not find specified file {}'.format(val_csv_fn))

    ############ Check other parameters ############

    n_input_sources = 1 if aux_input_source is None else 2
    
    if temp:
        if temp_loss == 'none':
            temp_loss = None
            lambda_temp = 0.
        if temp_align_loss == 'none':
            temp_align_loss = None
            lambda_temp_align = 0.
    else:
        temp_loss = None
        temp_align_loss = None
        lambda_temp = 0.
        lambda_temp_align = 0.
             
    if temp:
        if model_arch not in ['RecurrentUnet', 'GRUUnet', 'NonRecurrentUnet']:
            print('Warning: no valid model architecture has been specified, we will use a RecurrentUnet.')
            model_arch = 'RecurrentUnet'
        if common_input_bands is not None:
            if common_input_bands == 1:
                print('Converting all input time series to grayscale')
            elif common_input_bands == 3:
                print('Converting all input time series to RGB')
            else:
                raise ValueError('common_input_bands should be None, 1 (grayscale) or 3 (RGB)')
        else:
            print('Keeping original bands in the input time series')
        if reverse:
            print('Input time series will be fed to the model in reverse order')
    else:
        model_arch = 'Unet'
    
    

    if len(n_negative_samples) != len(negative_sampling_schedule):
        raise ValueError('n_negative_samples and negative_sampling_schedule should have the same number of elements')
    control_training_set = len(n_negative_samples) > 0

    
    if undersample_validation < 1:
        raise ValueError('undersample_validation factor should be greater than 1')
    if debug:
        undersample_validation = 4
        print('Debug mode: only 1/{}th of the validation set will be used'.format(undersample_validation))  
        undersample_training = undersample_training * 20
        print('Debug mode: only 1/{}th of the training set will be used'.format(undersample_training))  
      
    validate = [False] * (num_epochs - 1) + [True]  # always perform validation after the last epoch
    if not skip_validation:
        if validation_period > 0:
            validate[validation_period-1::validation_period] = \
                                                                    [True] * (num_epochs//validation_period)
        else:
            raise ValueError('The validation period should be greater than 0')   
        
    if not augment_vals:
        color_jitter = 0
        gauss_blur_sigma = 0 

    exp_utils = ExpUtils(main_input_source,
                         aux_input_source, 
                         multitemp_eval=temp,
                         common_input_bands=common_input_bands,
                         augment_main_input=augment_vals,
                         jitter = color_jitter,
                         sigma_max = gauss_blur_sigma,
                         grayscale_prob=grayscale_prob,
                         normalize_temp_inputs=normalize_temp_inputs,
                         std_gray_noise=std_gray_noise)
    
    save_dict = {
            'args': args_dict,
            'train_losses': [],
            'train_total_losses': [],
            'model_checkpoints': [],
            'optimizer_checkpoints' : [],
            'scheduler_checkpoints' : [],
            'proportion_negative_samples' : [],
            'random_state': [],
        }
    if temp_loss is not None:
        save_dict['train_temp_losses'] = []
    if temp_align_loss is not None:
        save_dict['train_temp_align_losses'] = []
    if not skip_validation:
        save_dict['val_reports'] = []
        save_dict['val_cms'] = []
        save_dict['val_epochs'] = []
        save_dict['val_losses'] = []
        if temp_loss is not None:
            save_dict['val_temp_losses'] = []
        if temp_align_loss is not None:
            save_dict['val_temp_align_losses'] = []

    # device = torch.device('cpu')
    if torch.cuda.is_available():
            device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available")

    print(args_dict)
    
    seed = random_seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    
    ############ Setup model ##################################################

    # Set model architecture
    decoder_channels = (256, 128, 64, 32)
    upsample = (True, True, True, False)
    if n_input_sources > 1:
        # 2 input modalities
        aux_in_channels = exp_utils.input_channels['input_aux']
        aux_in_position = 0
    else:
        # 1 input modality
        aux_in_channels = None
        aux_in_position = None
    init_stride = [1, 1]

    # Create model and criterion, and forward + backward pass function
    
    if resume_training:
        # load the state dicts
        if starting_model_fn is None:
            starting_fn = model_fn
        else:
            starting_fn = starting_model_fn
        starting_point = torch.load(starting_fn)
        try:
            starting_point_model_arch = starting_point['model_params']['model_arch']
            if starting_point_model_arch != model_arch:
                print("Warning: the starting point's architecture is not the same as the one specified in the "
                      "arguments. We will use the architecture of the arguments") 
        except KeyError:
            pass
            
    if temp:
        if model_arch == 'RecurrentUnet':
            print('Training a RecurrentUnet')
            model = RecurrentUnet(
                        encoder_depth=4, 
                        decoder_channels=decoder_channels,
                        in_channels=exp_utils.input_channels['input_main'], 
                        out_channels=exp_utils.output_channels,
                        upsample=upsample,
                        aux_in_channels=aux_in_channels,
                        aux_in_position=aux_in_position,
                        init_stride=init_stride,
                        bn_momentum=bn_momentum,
                        reverse=reverse)
        elif model_arch == 'GRUUnet':
            print('Training a GRUUnet')
            if not gru_irreg:
                gru_norm_dt = False # just for clarity
            model = GRUUnet(encoder_depth=4, 
                            decoder_channels=decoder_channels,
                            in_channels=exp_utils.input_channels['input_main'], 
                            out_channels=exp_utils.output_channels,
                            upsample=upsample,
                            aux_in_channels=aux_in_channels,
                            aux_in_position=aux_in_position,
                            init_stride=init_stride,
                            bn_momentum=bn_momentum,
                            reverse=reverse,
                            unet_out_channels=exp_utils.output_channels,
                            gru_irreg=gru_irreg,
                            gru_reset_channels=1,
                            gru_update_channels=1,
                            gru_kernel_size=gru_kernel_size,
                            gru_last_actv=gru_last_actv,
                            gru_init=gru_init,
                            gru_input=gru_input,
                            gru_norm_dt=gru_norm_dt)
        else:
            print('Using a mono-temporal Unet')
            model = NonRecurrentUnet(encoder_depth=4, 
                        decoder_channels=decoder_channels,
                        in_channels = exp_utils.input_channels['input_main'], 
                        out_channels = exp_utils.output_channels,
                        upsample = upsample,
                        aux_in_channels = aux_in_channels,
                        aux_in_position = aux_in_position,
                        init_stride=init_stride,
                        bn_momentum=bn_momentum)
        fit = utils.fit_temp
    else:
        print('Training a Unet')
        model = Unet(
                    encoder_depth=4, 
                    decoder_channels=decoder_channels,
                    in_channels=exp_utils.input_channels['input_main'], 
                    out_channels=exp_utils.output_channels,
                    upsample=upsample,
                    aux_in_channels=aux_in_channels,
                    aux_in_position=aux_in_position,
                    init_stride=init_stride,
                    bn_momentum=bn_momentum)
        fit = utils.fit

    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(num_train_params))
    model = model.to(device)
    
    if exp_utils.n_classes == 2:
        if model_arch == 'GRUUnet':
            seg_criterion = MyBCELoss(ignore_val=exp_utils.i_nodata_val)
        else:
            seg_criterion = MyBCEWithLogitsLoss(ignore_val=exp_utils.i_nodata_val)
    else:
        weights = torch.FloatTensor(exp_utils.get_CE_weights(exp_utils.class_freq['seg'], 1, 0))
        print('Loss weights: {}'.format(weights))
        seg_criterion = nn.CrossEntropyLoss(reduction = 'mean', 
                                            ignore_index=exp_utils.i_nodata_val, 
                                            weight=weights.to(device))
    if not skip_validation:
        val_seg_criterion = seg_criterion

    if temp_loss is not None:
        if temp_loss == 'MSE':
            temp_criterion = MyTemporalMSELoss(ignore_val=exp_utils.i_nodata_val, 
                                                seg_normalization=model.seg_normalization,
                                                use_temp_weights=weight_temp_loss)
        elif temp_loss == 'CE':
            if exp_utils.n_classes == 2:
                tempCE_seg_criterion = MyBCELoss(ignore_val=None)
            else:
               raise NotImplementedError
            temp_criterion = MyTemporalCELoss(decision_func=exp_utils.decision_func, 
                                              seg_criterion=tempCE_seg_criterion,
                                              ignore_val=exp_utils.i_nodata_val,
                                              seg_normalization=model.seg_normalization,
                                              use_temp_weights=weight_temp_loss)
        else:
            raise NotImplementedError('{} for temporal consistency loss not implemented'.format(temp_loss))
    else:
        temp_criterion = None
    if temp_align_loss is not None:
        if temp_align_loss == 'graddot':
            temp_align_criterion = MyGradDotTemporalLoss(device, 
                                    ignore_val=exp_utils.i_nodata_val, 
                                    seg_normalization=model.seg_normalization,
                                    use_temp_weights=weight_temp_loss,
                                    scale_by_norm=scale_by_norm,
                                    asymmetrical=asym_align)
            
        elif temp_align_loss == 'gradnorm':
            temp_align_criterion = MyGradNormTemporalLoss(device, 
                                                    ignore_val=exp_utils.i_nodata_val, 
                                                    seg_normalization=model.seg_normalization,
                                                    use_temp_weights=weight_temp_loss)
        else:
            raise NotImplementedError('{} for temporal alignment loss not implemented'.format(temp_loss))
    else:
        temp_align_criterion = None

    if isinstance(model, (Unet, RecurrentUnet, NonRecurrentUnet)):
        optimizer = optim.AdamW(model.parameters(), lr=lr_fe, amsgrad=True)
    else:
        optimizer = optim.AdamW([
                                {'name': 'fe', 'params': model.unet.parameters(), 'lr': lr_fe},
                                {'name': 'temp', 'params': model.gru.parameters(), 'lr': lr_temp}
                                ], lr = lr_fe, amsgrad=True)
    
    try:    
        print('Initial learning rate: {}'.format([(pg['name'], pg['lr']) for pg in optimizer.param_groups]))
    except KeyError:
        print('Initial learning rate: {}'.format([pg['lr'] for pg in optimizer.param_groups]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min',
                                                     factor=0.1,
                                                     patience=2,
                                                     threshold=1e-4,
                                                     verbose=True)

    # load checkpoints if resuming training from existing model
    if resume_training:
        try:
            if isinstance(model, (NonRecurrentUnet, RecurrentUnet, GRUUnet)) and starting_model_fn is not None: 
                try:
                    model.unet.load_state_dict(starting_point['model']) 
                except RuntimeError:
                    model.load_state_dict(starting_point['model'])
            else:
                model.load_state_dict(starting_point['model'])
                
        except RuntimeError:
            pretrained_model_dict =  starting_point['model']
            if isinstance(model, (NonRecurrentUnet, RecurrentUnet, GRUUnet)) and starting_model_fn is not None: 
                new_model_dict = model.unet.state_dict()
            else:
                new_model_dict = model.state_dict()
            
            pretrained_only = {k: v for k, v in pretrained_model_dict.items() if k not in new_model_dict}
            new_only = {k: v for k, v in new_model_dict.items() if k not in pretrained_model_dict}
            if len(pretrained_only) > 0:
                print('Parameter names found in the starting point only:{} '.format(pretrained_only.keys()))
            if len(new_only) > 0:
                print('Parameter names found in the current model only: {}'.format(new_only.keys()))
            
            common_dict = {}
            mismatch_param_names = []
            for k, old_val in pretrained_model_dict.items():
                if k in new_model_dict:
                    new_shape = new_model_dict[k].shape
                    old_shape = old_val.shape
                    if old_shape != new_shape:
                        shape_diff = np.array([s1 - s2 for s1, s2 in zip(old_shape, new_shape)])
                        if np.any(shape_diff < 0):
                            if np.all(shape_diff <= 0):
                                common_dict[k] = torch.zeros_like(new_model_dict[k]) 
                                common_dict[k][:, :3] = old_val[:, :3] 
                                common_dict[k][:, -1] = old_val[:, -1]
                            else:
                                raise RuntimeError('Shape mismatch not supported: {} in pretrained model, {} in '
                                                   'current model for parameter {}'.format(old_shape, new_shape, k))
                        elif np.any(shape_diff > 0):
                            if np.all(shape_diff >= 0):
                                common_dict[k] = old_val[[slice(s) for s in new_shape]]
                            else:
                                raise RuntimeError('Shape mismatch not supported: {} in pretrained model, {} in '
                                                   'current model for parameter {}'.format(old_shape, new_shape, k))
                        mismatch_param_names.append(k)
                    else:
                        common_dict[k] = old_val
                                    
            # overwrite entries in the existing state dict
            new_model_dict.update(common_dict) 
            
            # load the new state dict
            if isinstance(model, (NonRecurrentUnet, RecurrentUnet, GRUUnet)) and starting_model_fn is not None: 
                model.unet.load_state_dict(new_model_dict) 
            else:
                model.load_state_dict(new_model_dict)
            
            if freeze_matching_params > 0:
                
                if isinstance(model, (NonRecurrentUnet, RecurrentUnet, GRUUnet)) and starting_model_fn is not None:
                    print('For the first {} epochs, only parameters {} in the Unet will be trained.'.format(
                                                                                        freeze_matching_params, 
                                                                                        mismatch_param_names))
                    for name, p in model.unet.named_parameters():
                        if name not in mismatch_param_names:
                            p.requires_grad = False
                else:
                    print('For the first {} epochs, only parameters {} will be trained.'.format(
                                                                                        freeze_matching_params, 
                                                                                        mismatch_param_names))
                    for name, p in model.named_parameters():
                        if name not in mismatch_param_names:
                            p.requires_grad = False
        else:
            if isinstance(model, (NonRecurrentUnet, RecurrentUnet, GRUUnet)) \
            and starting_model_fn is not None \
            and freeze_matching_params > 0:
                print('For the first {} epochs, the parameters of the Unet will be frozen.'.format(
                                                                                        freeze_matching_params))
                for name, p in model.unet.named_parameters():
                    p.requires_grad = False
                
            
        
        for el in optimizer.param_groups:
            try:
                if el['name'] == 'fe':
                    el['lr'] = lr_fe
                elif el['name'] == 'temp':
                    el['lr'] = lr_temp
            except KeyError:
                el['lr'] = lr_fe
        # set the starting epoch
        if new_history:
            starting_epoch = 0
        else:
            optimizer.load_state_dict(starting_point['optimizer'])
            scheduler.load_state_dict(starting_point['scheduler'])
            starting_epoch = starting_point['epoch'] + 1
            # set the random state of when the pretraining was stopped
            try:
                random.setstate(starting_point['random_state']['random'])
                np.random.set_state(starting_point['random_state']['numpy'])
                torch.set_rng_state(starting_point['random_state']['pytorch'])
            except KeyError:
                pass
    else:
        starting_epoch = 0
        
    ############ Setup data ###################################################
    
    print('Creating dataset...')
    tic = time.time()
    
    # create dataset
    if temp:
        dataset = TempTrainingDataset(
                                                    dataset_csv=train_csv_fn,
                                                    n_input_sources=n_input_sources,
                                                    exp_utils = exp_utils,
                                                    control_training_set=control_training_set,
                                                    n_neg_samples = None,
                                                    patch_size=patch_size,
                                                    num_patches_per_tile = num_patches_per_tile,
                                                    verbose=False,
                                                    undersample=undersample_training,
                                                    augment_flip=augment_flip,
                                                    )
    else:
        dataset = TrainingDataset( 
                                                    dataset_csv=train_csv_fn,
                                                    n_input_sources=n_input_sources,
                                                    exp_utils = exp_utils,
                                                    control_training_set=control_training_set,
                                                    n_neg_samples = None,
                                                    patch_size=patch_size,
                                                    num_patches_per_tile = num_patches_per_tile,
                                                    verbose=False,
                                                    undersample=undersample_training,
                                                    augment_flip=augment_flip,
                                                    )

    # create array containing the number of negatives samples to be selected for each epoch
    n_neg_samples = np.full(num_epochs, dataset.n_negatives)
    if control_training_set:
        n_controlled_epochs = min(num_epochs, np.sum(negative_sampling_schedule))
        n_neg_samples[:n_controlled_epochs] = np.repeat(
                                                n_negative_samples, 
                                                negative_sampling_schedule
                                                        )[:n_controlled_epochs]
        # clip the array to the total number of negative samples in the dataset
        n_neg_samples[:n_controlled_epochs] = np.minimum(n_neg_samples[:n_controlled_epochs], dataset.n_negatives)

    print("finished in %0.4f seconds" % (time.time() - tic))
    
    g = torch.Generator()
    if resume_training:
        try:
            g.set_state(starting_point['random_state']['pytorch_generator'])
        except KeyError:
            print('Random state of the starting point not found. Manually fixing the random seed to {}.'.format(seed))
            g.manual_seed(seed)
    else:
        g.manual_seed(seed)
        
    if temp:
        collate_fn = lambda x: collate_variable_length_series(x, pad_val=exp_utils.i_nodata_val)
    else:
        collate_fn = None
        
    # create dataloader
    print('Creating dataloader...')
    tic = time.time()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers_train,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=dataset.seed_worker,
        generator=g,
    )
    print('batch_size: {}'.format(batch_size))
    print("finished in %0.4f seconds" % (time.time() - tic))
        
    ######################### VALIDATION SETUP ################################

    if not skip_validation:
        infer_patch_size = 1000 #256 # 
        infer_padding = 0 #64 # 
         # the value below might not be attained because one batch can only contain patches from the same tile
        infer_batch_size = 1
        if temp:
            inference = utils.TempInference(model, 
                    val_csv_fn, 
                    exp_utils, 
                    output_dir=None, 
                    evaluate=True, 
                    save_hard=False, 
                    save_soft=False,
                    save_temp_diff=False,
                    batch_size=infer_batch_size, 
                    patch_size=infer_patch_size, 
                    padding=infer_padding,
                    num_workers=num_workers_val, 
                    device=device,
                    undersample=undersample_validation, 
                    random_seed=random_seed) 
        else:
            inference = utils.Inference(model, 
                    val_csv_fn, 
                    exp_utils, 
                    output_dir=None, 
                    evaluate=True, 
                    save_hard=False, 
                    save_soft=False,
                    batch_size=infer_batch_size, 
                    patch_size=infer_patch_size, 
                    padding=infer_padding, 
                    num_workers=num_workers_val, 
                    device=device,
                    undersample=undersample_validation)

    
    ############ Training #####################################################
    print('Starting training') 
    n_batches_per_epoch = int(len(dataset.fns) * num_patches_per_tile / batch_size)

    for i, epoch in enumerate(range(starting_epoch, starting_epoch + num_epochs)):       
            
        print('\nTraining epoch: {}'.format(epoch))
        if control_training_set or undersample_training > 1:
            # update the dataset to select the right number of random negative samples
            dataset.select_tiles(n_neg_samples[i])     
            if n_neg_samples[i] != n_neg_samples[i-1] or i==0:
                # recalculate the number of batches per epoch (for the progress bar)
                n_batches_per_epoch = int(len(dataset.fns) * num_patches_per_tile / batch_size) 
                    
        if resume_training:
            if i > 0:
                if freeze_matching_params == i :           
                    print('All the model parameters will be trained from now.')
                    for name, p in model.named_parameters():
                        p.requires_grad = True
        
        # shuffle data at every epoch (placed here so that all the workers use the same permutation)
        # debug
        # print('SHUFFLE DEACTIVATED!')
        dataset.shuffle()

        # forward and backward pass
        training_loss = fit(
                            model=model,
                            device=device,
                            dataloader=dataloader,
                            optimizer=optimizer,
                            n_batches=n_batches_per_epoch,
                            seg_criterion=seg_criterion,
                            temp_criterion=temp_criterion,
                            temp_align_criterion=temp_align_criterion,
                            lambda_temp=lambda_temp,
                            lambda_temp_align=lambda_temp_align,
                            update_period=update_period,
                            seg_eval_year=exp_utils.tlm_target_year
           )
        # debug 
        # training_loss = (0, 0, 0)

        # evaluation (validation) 
        if validate[i]:
            print('Validation')
            results = inference.infer(val_seg_criterion, 
                                        temp_criterion,
                                        temp_align_criterion)
            cm, report, val_losses = results
            # collect individual validation losses and compute total validation loss
            if temp:
                val_seg_loss, *other_losses = val_losses
            else:
                val_seg_loss = val_losses
            val_total_loss = val_seg_loss
            if temp_loss is not None:
                val_temp_loss_per_year, val_temp_loss, *_ = other_losses
                val_total_loss += lambda_temp * val_temp_loss
            if temp_align_loss is not None:
                _, _, val_temp_align_loss_per_year, val_temp_align_loss = other_losses
                val_total_loss += lambda_temp_align * val_temp_align_loss
        
        # WARNING: if validation is not done at every epoch, the scheduler will see the same validation loss several times            
        scheduler.step(val_total_loss)
    
        # update and save dictionary containing metrics
        if control_training_set:
            save_dict['proportion_negative_samples'].append(n_neg_samples[i]/dataset.n_fns_all)
        else:
            save_dict['proportion_negative_samples'].append(NaN)    
        save_dict['args']['num_epochs'] = epoch + 1 # number of epochs already computed

        # store training losses
        training_total_loss, training_seg_loss, *other = training_loss            
        save_dict['train_total_losses'].append(training_total_loss)        
        save_dict['train_losses'].append(training_seg_loss)
        if temp_loss is not None:
            training_temp_loss, *_ = other
            save_dict['train_temp_losses'].append(training_temp_loss)
        if temp_align_loss is not None:
            _, training_temp_align_loss = other
            save_dict['train_temp_align_losses'].append(training_temp_align_loss)
        
        # store validation losses/metrics
        if validate[i]: 
            save_dict['val_reports'].append(report)
            save_dict['val_cms'].append(deepcopy(cm)) # deepcopy is necessary
            save_dict['val_epochs'].append(epoch)
            save_dict['val_losses'].append(val_seg_loss)
            if temp_loss is not None:
                save_dict['val_temp_losses'].append(val_temp_loss_per_year)
            if temp_align_loss is not None:
                save_dict['val_temp_align_losses'].append(val_temp_align_loss_per_year)
                
        with open(log_fn, 'wb') as f:
            torch.save(save_dict, f)

        # save last checkpoint in a separate file with parameters necessary to 
        # instantiate model for inference or resume training
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        torch_random_state = torch.get_rng_state()
        torch_generator_random_state = g.get_state()
        random_state_dict = {'random': random_state,
                                     'numpy': np_random_state,
                                     'pytorch': torch_random_state,
                                     'pytorch_generator': torch_generator_random_state}
        last_checkpoint = {'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch,
                            'model_params': {'model_arch': model_arch,
                                             'reverse': reverse,
                                             'common_input_bands': common_input_bands,
                                             }, 
                            'random_state': random_state_dict}
        if model_arch == 'GRUUnet':
            last_checkpoint['model_params']['gru_irreg'] = gru_irreg
            last_checkpoint['model_params']['gru_kernel_size'] = gru_kernel_size
            last_checkpoint['model_params']['gru_input'] = gru_input
            last_checkpoint['model_params']['gru_last_actv'] = gru_last_actv
            last_checkpoint['model_params']['gru_init'] = gru_init
            last_checkpoint['model_params']['gru_norm_dt'] = gru_norm_dt
        torch.save(last_checkpoint, model_fn.replace('model.pt', 'model_epoch{}.pt'.format(epoch)))
    
    if not skip_validation:
        inference.end()  

########################################################################################################################

if __name__ == "__main__":
    
    debug = True #
    # run parameters
    random_seed = 0
    if debug:
        exp_name = 'debug'
    else:
        exp_name = 'new_experiment' 
    output_dir = os.path.join('output', exp_name)
    temp = True
    num_epochs = 1 #20 #30
    validation_period = 1
    
    # data loading
    undersample_validation = 1 
    negative_sampling_schedule = [num_epochs]
    
    # training parameters
    patch_size = 128 # in meters
    
    # data pre-processing
    common_input_bands = None
    
    # resource allocation
    num_workers_train = 0 #8 
    num_workers_val = 0 #4
    
    # misc
    no_user_input = True
    
    # parameters for multi-temporal training
    if temp:
        # run parameters
        main_input_source = 'SItemp' 
        aux_input_source = 'ALTI' 
        train_csv_fn = 'data/csv/{}100cm_1946_to_2020_{}100cm_TLM6c_train_with_counts.csv'.format(main_input_source, 
                                                                                        aux_input_source)
        val_csv_fn = 'data/csv/{}100cm_1946_to_2020_{}100cm_TLM6c_multitemp_mylabels_val.csv'.format(main_input_source, 
                                                                                            aux_input_source)
        
        new_history = True
        starting_model_name = 'Unet_SI2020_100cm_grayaugment_rs0'
        starting_model_fn =  os.path.join('output', 
                                            starting_model_name, 
                                            'training', 
                                            '{}_model_epoch19.pt'.format(starting_model_name))
        resume_training = True # must be True to train GRUUnet
        freeze_matching_params = 0 #num_epochs
        model_arch = 'GRUUnet' #'NonRecurrentUnet' #
        
        # data loading
        undersample_training = 1 # value > 1 reduces data loading cost
        num_patches_per_tile = 4
        n_negative_samples = [20//undersample_training]  

        # training parameters  
        lr_fe = 1e-5 
        lr_temp = 1e-3
        batch_size = 8 
        update_period = 256 // batch_size # 64 // batch_size # simulate a larger batch size
        bn_momentum = 1e-5 #0.001
        
        # data augmentation
        augment_vals = True 
        gauss_blur_sigma = 0.25
        color_jitter = 0.25
        grayscale_prob = 0
        std_gray_noise = 0
        
        # loss
        lambda_temp = 1. #1 # 20
        temp_loss = 'CE' #'MSE' # 'none' #'graddot'
        lambda_temp_align = 1.
        temp_align_loss = 'gradnorm' #'graddot'
        scale_by_norm=True,
        asym_align = False
        weight_temp_loss = True
        
        # temporal model
        reverse = True
        gru_irreg = True
        gru_kernel_size = 7
        gru_input = 'df'
        gru_norm_dt = False
        
        train(output_dir=output_dir,
                main_input_source=main_input_source,
                aux_input_source=aux_input_source,
                train_csv_fn=train_csv_fn,
                val_csv_fn=val_csv_fn,
                temp=temp,
                num_epochs=num_epochs,
                random_seed=random_seed,
                new_history=new_history,
                starting_model_fn=starting_model_fn,
                resume_training=resume_training,
                freeze_matching_params=freeze_matching_params,
                model_arch=model_arch,
                undersample_training=undersample_training,
                undersample_validation=undersample_validation,
                num_patches_per_tile=num_patches_per_tile,
                n_negative_samples=n_negative_samples,
                batch_size=batch_size,
                patch_size=patch_size,
                lr_fe=lr_fe,
                lr_temp=lr_temp,
                update_period=update_period,
                bn_momentum=bn_momentum,
                augment_vals=augment_vals,
                gauss_blur_sigma=gauss_blur_sigma,
                color_jitter=color_jitter,
                grayscale_prob=grayscale_prob,
                std_gray_noise=std_gray_noise,
                lambda_temp=lambda_temp,
                temp_loss=temp_loss,
                lambda_temp_align = lambda_temp_align,
                temp_align_loss = temp_align_loss,
                scale_by_norm=scale_by_norm,
                asym_align=asym_align,
                weight_temp_loss=weight_temp_loss,
                reverse=reverse,
                gru_irreg=gru_irreg,
                gru_kernel_size=gru_kernel_size,
                gru_input=gru_input,
                gru_norm_dt = gru_norm_dt,
                common_input_bands=common_input_bands,
                num_workers_train=num_workers_train,
                num_workers_val=num_workers_val,
                debug=debug,
                no_user_input=no_user_input)
        
    # parameters for mono-temporal training    
    else:
        # run parameters
        main_input_source = 'SI2020'
        aux_input_source = 'ALTI'  
        train_csv_fn = 'data/csv/{}_{}_TLM6c_train_with_counts.csv'.format(main_input_source, 
                                                                            aux_input_source)
        val_csv_fn = 'data/csv/{}_{}_TLM6c_val.csv'.format(main_input_source, 
                                                            aux_input_source)
        resume_training = False
        if resume_training:
            starting_model_fn = 'output/Unet_SI2020_100cm_grayaugment_rs0/training/Unet_SI2020_100cm_grayaugment_rs0_model_epoch19.pt'
            new_history = False
        else:
            starting_model_fn = None
            new_history = True
        model_arch = 'Unet'
        
        # data loading
        undersample_training = 1
        num_patches_per_tile = 2 #small value to keep a diverse batch #(1000/128)^2 = 61 
        n_negative_samples = [20//undersample_training] 
        
        # training parameters
        lr_fe = 1e-4
        batch_size = 32 
        update_period = 1
        bn_momentum = 0.1
        
        # data augmentation
        augment_vals = True
        gauss_blur_sigma = 0.5
        color_jitter = 0.5
        grayscale_prob = 0.5
        std_gray_noise = 0.1
        
        train(output_dir=output_dir,
                main_input_source=main_input_source,
                aux_input_source=aux_input_source,
                train_csv_fn=train_csv_fn,
                val_csv_fn=val_csv_fn,
                temp=temp,
                num_epochs=num_epochs,
                random_seed=random_seed,
                new_history=new_history,
                starting_model_fn=starting_model_fn,
                resume_training=resume_training,
                model_arch=model_arch,
                undersample_training=undersample_training,
                undersample_validation=undersample_validation,
                num_patches_per_tile=num_patches_per_tile,
                n_negative_samples=n_negative_samples,
                batch_size=batch_size,
                patch_size=patch_size,
                lr_fe=lr_fe,
                update_period=update_period,
                bn_momentum=bn_momentum,
                augment_vals=augment_vals,
                gauss_blur_sigma=gauss_blur_sigma,
                color_jitter=color_jitter,
                grayscale_prob=grayscale_prob,
                std_gray_noise=std_gray_noise,
                common_input_bands=common_input_bands,
                num_workers_train=num_workers_train,
                num_workers_val=num_workers_val,
                debug=debug,
                no_user_input=no_user_input)
        