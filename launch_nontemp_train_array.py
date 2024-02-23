# launch a series of features extractor pre-training experiments

import os
from train import train

temp = False

# run parameters
num_epochs = 20
main_input_source = 'SI2020' 
aux_input_source = 'ALTI' 
train_csv_fn = 'data/csv/{}_{}_TLM6c_train_with_counts.csv'.format(main_input_source, 
                                                                    aux_input_source)
val_csv_fn = 'data/csv/{}_{}_TLM6c_val.csv'.format(main_input_source, 
                                                    aux_input_source)
model_arch = 'Unet' #'NonRecurrentUnet' #
random_seed = 0
validation_period = 1

# data loading
undersample_validation = 1
undersample_training = 1
num_patches_per_tile = 2
n_negative_samples = [20//undersample_training]
negative_sampling_schedule = [num_epochs]

# training parameters
patch_size = 128 # in meters
lr_fe = 1e-4
# each batch will containg patches extracted from batch_size/num_patches_per_tile different tiles. This impacts the
# batch norm statistics.
batch_size = 32 
update_period = 1
bn_momentum = 0.1

# data augmentation
augment_vals = True 
gauss_blur_sigma = 0.5
color_jitter = 0.5
grayscale_prob = 0.5

# data pre-processing
common_input_bands = None

# resource allocation
num_workers_train = 8
num_workers_val = 4

# misc
no_user_input = True
debug=False
    
exp_name_base = 'Unet_SI2020_100cm_grayaugment'

for exp_name_base, common_input_bands, grayscale_prob in \
    [('Unet_SI2020_100cm_grayaugment', None, 0.5),
     ('Unet_SI2020gray_100cm', 1, 0),
     ('Unet_SI2020_100cm_noaugment', None, 0.0)]:

    for random_seed in range(5):
        exp_name = '{}_rs{}'.format(exp_name_base, random_seed)
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
                model_arch=model_arch,
                undersample_training=undersample_training,
                undersample_validation=undersample_validation,
                num_patches_per_tile=num_patches_per_tile,
                n_negative_samples=n_negative_samples,
                negative_sampling_schedule=negative_sampling_schedule,
                batch_size=batch_size,
                patch_size=patch_size,
                lr_fe=lr_fe,
                bn_momentum=bn_momentum,
                update_period=update_period,
                augment_vals=augment_vals,
                gauss_blur_sigma=gauss_blur_sigma,
                color_jitter=color_jitter,
                grayscale_prob=grayscale_prob,
                common_input_bands=common_input_bands,
                num_workers_train=num_workers_train,
                num_workers_val=num_workers_val,
                debug=debug,
                no_user_input=no_user_input)

