# launch a series of multitemporal fine-tuning experiments

import os
from train import train

temp = True

# run parameters
num_epochs = 40
main_input_source = 'SItemp' 
aux_input_source = 'ALTI' 
train_csv_fn = 'data/csv/{}_1946_to_2020_{}_TLM6c_train_with_counts.csv'.format(main_input_source, 
                                                                                aux_input_source)
val_csv_fn = 'data/csv/{}_1946_to_2020_{}_TLM6c_multitemp_mylabels_val.csv'.format(main_input_source, 
                                                                                    aux_input_source)

# define pre-trained feature extractor. Choose the correct block of code below
# if trained locally, uncomment following lines
starting_model_name = 'Unet_SI2020_100cm_grayaugment_rs0'
starting_model_fn =  os.path.join('output', 
                                    starting_model_name, 
                                    'training', 
                                    '{}_model.pt'.format(starting_model_name))
# if downloaded from the cloud, uncomment following lines
# starting_model_name = 'Unet_randseed0'
# starting_model_fn =  os.path.join('trainedModels', '{}_model.pt'.format(starting_model_name))

resume_training = True 
model_arch = 'GRUUnet' 
random_seed = 0
validation_period = 1

# data loading
undersample_validation = 1
undersample_training = 1
num_patches_per_tile = 4
n_negative_samples = [20//undersample_training]
negative_sampling_schedule = [num_epochs]

# training parameters
patch_size = 128 # in meters
lr_fe = 1e-6
lr_temp = 1e-4
# each batch will containg patches extracted from batch_size/num_patches_per_tile different tiles. This impacts the
# batch norm statistics.
batch_size = 8 
update_period = 256 // batch_size # simulate a larger batch size. 
bn_momentum = 1e-5

# data augmentation
augment_vals = True 
gauss_blur_sigma = 0.25
color_jitter = 0.25
grayscale_prob = 0

# loss
lambda_temp = 1.0
temp_loss = 'CE' #MSE'#'none' 
lambda_temp_align = 1.0
temp_align_loss = 'CA' #'CA_ablation' #'none
scale_by_norm = True
asym_align = True
weight_temp_loss = True

# temporal model
reverse = True
gru_irreg = True # False
gru_kernel_size = 7

# data pre-processing
common_input_bands = None

# resource allocation
num_workers_train = 8
num_workers_val = 4

# misc
no_user_input = True
debug=False

for random_seed in range(5):
    if model_arch == 'GRUUnet':
        if gru_irreg:
            model_arch_name = 'IrregGRU'
        else:
            model_arch_name = 'GRU'
    elif model_arch == 'NonRecurrentUnet':
        model_arch_name = 'NRUnet'
    else:
        raise ValueError
    
    if temp_align_loss=='none':
        temp_loss_name = temp_loss
        lambda_temp_name = lambda_temp
    else:
        if not scale_by_norm:
            temp_loss_name = '_'.join([temp_loss, 'unscaled{}'.format(temp_align_loss)])
        else:
            if asym_align:
                temp_loss_name = '_'.join([temp_loss, 'asym{}'.format(temp_align_loss)])
            else:
                temp_loss_name = '_'.join([temp_loss, temp_align_loss])
        lambda_temp_name = '_'.join([str(lambda_temp), str(lambda_temp_align)])

    exp_name = '{}{}_{}_lrfe{}_lrtemp{}_tloss{}_ltemp{}_rs{}'.format(
                                model_arch_name,
                                gru_kernel_size,
                                'bwrd' if reverse else 'fwrd',
                                lr_fe,
                                lr_temp,
                                temp_loss_name,
                                lambda_temp_name,
                                random_seed)
    exp_name = exp_name.replace('.', '_').replace('-', 'm')
    output_dir = os.path.join('output', exp_name)
    
    print('Launching experiment {}'.format(exp_name))

    train(  output_dir=output_dir,
            main_input_source=main_input_source,
            aux_input_source=aux_input_source,
            train_csv_fn=train_csv_fn,
            val_csv_fn=val_csv_fn,
            temp=temp,
            num_epochs=num_epochs,
            random_seed=random_seed,
            starting_model_fn=starting_model_fn,
            resume_training=resume_training,
            model_arch=model_arch,
            undersample_training=undersample_training,
            undersample_validation=undersample_validation,
            num_patches_per_tile=num_patches_per_tile,
            n_negative_samples=n_negative_samples,
            negative_sampling_schedule=negative_sampling_schedule,
            batch_size=batch_size,
            patch_size=patch_size,
            lr_fe=lr_fe,
            lr_temp=lr_temp,
            bn_momentum=bn_momentum,
            update_period=update_period,
            augment_vals=augment_vals,
            gauss_blur_sigma=gauss_blur_sigma,
            color_jitter=color_jitter,
            grayscale_prob=grayscale_prob,
            lambda_temp=lambda_temp,
            temp_loss=temp_loss,
            lambda_temp_align=lambda_temp_align,
            temp_align_loss=temp_align_loss,
            scale_by_norm=scale_by_norm,
            asym_align=asym_align,
            weight_temp_loss=weight_temp_loss,
            reverse=reverse,
            gru_irreg=gru_irreg,
            gru_kernel_size=gru_kernel_size,
            common_input_bands=common_input_bands,
            num_workers_train=num_workers_train,
            num_workers_val=num_workers_val,
            debug=debug,
            no_user_input=no_user_input)

