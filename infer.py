import os
import torch
from models import Unet, NonRecurrentUnet
from models.unet.model import GRUUnet
import utils
from utils import ExpUtils
import numpy as np
import random

def infer(csv_fn,
          model_fn,
          output_dir,
          main_input_source='SItemp',
          aux_input_source='ALTI',
          overwrite=False,
          evaluate=False, 
          save_hard=True,
          save_soft=True,
          save_temp_diff=False,
          temp=True,
          batch_size=4,
          patch_size=256,
          padding=64,
          random_seed=0,
          num_workers=4):
    
    """ 
    - csv_fn: str, path to the csv file containing the list of input and target tiles to use
    - model_fn: str, path to the trained model file
    - output_dir: str, path to the directory where the output will be written
    - main_input_source: str, name of the main input source
    - aux_input_source: str, name of the auxiliary input source
    - overwrite: bool, if True, the output directory will be overwritten if it already exists
    - evaluate: bool, if True, the model will be evaluated using the target_tiles in csv_fn
    - save_hard: bool, if True, the hard predictions will be saved
    - save_soft: bool, if True, the soft predictions will be saved
    - save_temp_diff: bool, if True, the temporal difference will be saved
    - temp: if True, the model processes multi-temporal data, if False, each tile and acquisition year is processed 
            independently
    - batch_size: int, batch size
    - patch_size: int, size of the patches to process, in pixels
    - padding: int, padding to add to the input tiles and to each patch, in pixels. Avoids border effects.
    - random_seed: int, random seed
    - num_workers: int, number of workers for the dataloader
    """
    
    args_dict = {'csv_fn': csv_fn,
                 'model_fn': model_fn,
                 'output_dir': output_dir,
                 'main_input_source': main_input_source,
                 'aux_input_source': aux_input_source,
                 'temp': temp,
                 'batch_size': batch_size,
                 'patch_size': patch_size,
                 'padding': padding,
                 'random_seed': random_seed,
                 'num_workers': num_workers}

    ############ Argument checking ###############

    # check paths of model and input
    if not os.path.exists(csv_fn):
        raise FileNotFoundError("{} does not exist".format(csv_fn))
    if not os.path.exists(model_fn):
        raise FileNotFoundError('{} does not exist.'.format(model_fn))

    # check output path
    if save_hard or save_soft or save_temp_diff or evaluate:
        if os.path.exists(output_dir):
            if os.path.isfile(output_dir):
                raise NotADirectoryError("A file was passed as `--output_dir`, please pass a directory!")
            elif len(os.listdir(output_dir)) > 0:
                if overwrite:
                    print("WARNING: Output directory {} already exists, we might overwrite data in it!"
                            .format(output_dir))
                else:
                    raise FileExistsError("Output directory {} already exists and isn't empty."
                                            .format(output_dir))
        else:
            print("{} doesn't exist, creating it.".format(output_dir))
            os.makedirs(output_dir)   
        if evaluate:
            metrics_fn =  os.path.join(output_dir, '{}_metrics.pt'.format(exp_name))     
            print('Metrics will be written to {}'.format(metrics_fn))
    
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
             
    # check gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available")

    # Set up data and training parameters
    model_obj = torch.load(model_fn)
    try:
        model_arch = model_obj['model_params']['model_arch']
    except KeyError:
        if temp:
            print('Model architecture information was not found. We will try to load the parameters to a NonRecurrentUnet.')
            model_arch = 'NonRecurrentUnet'
        else:
            model_arch = 'Unet'
    try:
        common_input_bands = model_obj['model_params']['common_input_bands']
    except KeyError:
        common_input_bands = None
        print('parameter "common_input_bands" not found in saved model hyperparameters, '
                'setting it to "{}".'.format(common_input_bands))
        
    if temp:
        
        try:
            temp_loop = model_obj['model_params']['temp_loop']
        except KeyError:
            temp_loop = False
        if temp_loop:
            try:
                rec_init = model_obj['model_params']['rec_init']
            except KeyError:
                rec_init = 'zero'
                print('parameter "rec_init" not found in saved model hyperparameters, setting it to "{}".'.format(rec_init))
            try:
                rec_features_norm = model_obj['model_params']['rec_features_norm']
            except KeyError:
                rec_features_norm = 'batchnorm'
                print('parameter "rec_features_norm" not found in saved model hyperparameters, setting it to '
                      '"{}".'.format(rec_features_norm))
                
            try:
                rec_features_clamp = model_obj['model_params']['rec_features_clamp']
            except KeyError:
                rec_features_clamp = 'clamp'
                print('parameter "rec_features_clamp" not found in saved model hyperparameters, setting it to '
                      '"{}".'.format(rec_features_clamp))
                
        try:
            reverse = model_obj['model_params']['reverse']
        except KeyError:
            reverse = True
            print('parameter "reverse" not found in saved model hyperparameters, setting it to "{}".'.format(reverse))
        if reverse:
            print('Input time series will be fed to the model in reverse order')
        if model_arch == 'GRUUnet':
            try:
                gru_kernel_size = model_obj['model_params']['gru_kernel_size']
            except KeyError:
                gru_kernel_size = 7
                print('parameter "gru_kernel_size" not found in saved model hyperparameters, setting it to {}.'\
                        .format(gru_kernel_size))
            try:
                gru_input = model_obj['model_params']['gru_input']
            except KeyError:
                gru_input = 'df'
                print('parameter "gru_input" not found in saved model hyperparameters, setting it to {}.'\
                        .format(gru_input))
            try:
                gru_irreg = model_obj['model_params']['gru_irreg']
            except KeyError:
                gru_irreg = False 
                print('parameter "gru_irreg" not found in saved model hyperparameters, setting it to {}.'.\
                        format(gru_irreg))
                    
    else:
        rec_init = None
        reverse = False
        rec_features_norm = 'batchnorm'
        rec_features_clamp = None
            
    exp_utils = ExpUtils(main_input_source, 
                         aux_input_source,
                         multitemp_eval=temp,
                        common_input_bands=common_input_bands)

    ############ Setup model ###############
    
    # Set model architecture
    decoder_channels = (256, 128, 64, 32)
    upsample = (True, True, True, False)
    aux_in_channels = exp_utils.input_channels['input_aux']
        
    init_stride = [1, 1] # to keep the same spatial resolution as the input
    # Create model
    if temp:
        if model_arch == 'GRUUnet':
            print('Using a GRUUnet')
            model = GRUUnet(encoder_depth=4, 
                        decoder_channels=decoder_channels,
                        in_channels = exp_utils.input_channels['input_main'], 
                        out_channels = exp_utils.output_channels,
                        upsample = upsample,
                        aux_in_channels = aux_in_channels,
                        init_stride=init_stride,
                        reverse=reverse,
                        unet_out_channels=exp_utils.output_channels,
                        gru_irreg=gru_irreg,
                        gru_reset_channels=1,
                        gru_update_channels=1,
                        gru_kernel_size=gru_kernel_size,
                        gru_input=gru_input) 
        else:
            print('Using a mono-temporal Unet')
            model = NonRecurrentUnet(encoder_depth=4, 
                        decoder_channels=decoder_channels,
                        in_channels = exp_utils.input_channels['input_main'], 
                        out_channels = exp_utils.output_channels,
                        upsample = upsample,
                        aux_in_channels = aux_in_channels,
                        init_stride=init_stride)
            
    else:
        model = Unet(encoder_depth=4, 
                        decoder_channels=decoder_channels,
                        in_channels = exp_utils.input_channels['input_main'], 
                        out_channels = exp_utils.output_channels,
                        upsample = upsample,
                        aux_in_channels = aux_in_channels,
                        init_stride=init_stride)
    try:
        model.load_state_dict(model_obj['model'])
    except RuntimeError:
        model.unet.load_state_dict(model_obj['model'])
    
    model = model.to(device)

    ############ Inference ###############
    if temp:
        inference = utils.TempInference(model, 
                                        csv_fn, 
                                        exp_utils, 
                                        output_dir=output_dir, 
                                        evaluate=evaluate, 
                                        save_hard=save_hard, 
                                        save_soft=save_soft,
                                        save_temp_diff=save_temp_diff, 
                                        batch_size=batch_size, 
                                        patch_size=patch_size, 
                                        padding=padding,
                                        num_workers=num_workers, 
                                        device=device, 
                                        undersample=1, 
                                        random_seed=random_seed)
    else:
        inference = utils.Inference(model, 
                                    csv_fn, 
                                    exp_utils, 
                                    output_dir=output_dir, 
                                    evaluate=evaluate, 
                                    save_hard=save_hard, 
                                    save_soft=save_soft, 
                                    batch_size=batch_size, 
                                    patch_size=patch_size, 
                                    padding=padding,
                                    num_workers=num_workers, 
                                    device=device, 
                                    undersample=1, 
                                    random_seed=random_seed)

    result = inference.infer()

    ############ Evaluation ###############
    
    if evaluate:
        if result is not None:
            cm, report, _ = result
                
            # Save metrics to file
            d = {
                'args': args_dict,
                'reports': report,
                'cms': cm
            }    
            with open(metrics_fn, 'wb') as f:
                torch.save(d, f)
                
    inference.end()            

#################################################################################################################

if __name__ == "__main__":
    
    use_all_seeds = False
    
    # proposed model
    # exp_name_list = ['NIrregGRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossCE_asymgraddot_ltemp1_0_1_0_rs1']
    # exp_name_list = ['NIrregGRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossCE_asymgraddot_ltemp1_0_1_0'] 
    
    # ConvGRU v.s. IrregCongGRU
    # exp_name_list = ['NIrregGRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossCE_asymgraddot_ltemp1_0_1_0_rs1',
    #                  'GRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossCE_asymgraddot_ltemp1_0_1_0_rs0',
    #                 ]
    
    # exp_name_list = ['GRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossCE_asymgraddot_ltemp1_0_1_0',
    #                 'NIrregGRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossCE_asymgraddot_ltemp1_0_1_0']
    
    # compare loss functions
    # exp_name_list = [#'NIrregGRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossMSE_ltemp2_0_rs2',
    #          'NIrregGRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossCE_ltemp2_0_rs2',
    #          #'NIrregGRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossnone_asymgraddot_ltemp0_0_2_0_rs2',
    #             'NIrregGRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossMSE_asymgraddot_ltemp1_0_1_0_rs4',
    #             'NIrregGRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossCE_asymgraddot_ltemp1_0_1_0_rs1'
    #             ]
    # exp_name_list = ['NIrregGRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossMSE_ltemp2_0_rs2']
    
    # compare pre-trained mono-temporal feature extractors
    # exp_name_list = [
    #                 'Unet_SI2020_100cm_grayaugment',
    #                 # 'Unet_SI2020_100cm_noaugment',
    #                 # 'Unet_SI2020gray_100cm'
    #                  ]
    exp_name_list = [
                    'Unet_SI2020_100cm_grayaugment_rs0' ]
    
    # Non Recurrent Unet with multi-temporal fine-tuning
    # exp_name_list = ['NRUnet7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossCE_asymgraddot_ltemp1_0_1_0']
    # exp_name_list = ['NRUnet7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossCE_asymgraddot_ltemp1_0_1_0_rs2']
    
    # ablation
    # exp_name_list = ['NIrregGRU7df_bwrd_freeze0_lrfe1em06_lrtemp0_0001_tlossCE_asymgradnorm_ltemp1_0_1_0'] 
    
    if use_all_seeds:
        new_exp_name_list = []
        for exp_name in exp_name_list:
            new_exp_name_list += ['{}_rs{}'.format(exp_name, rs) for rs in range(5)]
        exp_name_list = new_exp_name_list
                        
    data_set = 'mylabels_test' 
    epoch = 19 #39 
    end_year = 2020
    padding = 64  #0 for evaluation, 64 for vizualisation
        
    for exp_name in exp_name_list:
        infer(
            csv_fn='data/csv/SItemp100cm_1946_to_{}_ALTI100cm_TLM6c_multitemp_{}.csv'.format(end_year,
                                                                                        data_set),
        model_fn = 'output/{0}/training/{0}_model_epoch{1}.pt'.format(exp_name, epoch),
        output_dir='output/{}/inference/epoch_{}/{}_debug'.format(exp_name, epoch, data_set),
        main_input_source = 'SItemp',
        aux_input_source = 'ALTI',
        overwrite=True,
        evaluate=True, 
        save_hard=False,
        save_soft=False,
        save_temp_diff=False,
        temp=True,
        batch_size=512, #32, # does not matter if patch_size == tile_size
        patch_size=256, #256, 256 - 2*32 = 192, 1000/4 + 2 * 64 = 378, 1000 for evaluation, 256 for vizualisation
        padding=padding,
        random_seed=0,
        num_workers=4
        )