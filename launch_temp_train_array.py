import os
from train import train

temp = True
wandb_tracking = True

# run parameters
num_epochs = 40
main_input_source = 'SItemp' 
aux_input_source = 'ALTI' 
train_csv_fn = 'data/csv/{}_1946_to_2020_{}_TLM6c_train_with_counts.csv'.format(main_input_source, 
                                                                                aux_input_source)
val_csv_fn = 'data/csv/{}_1946_to_2020_{}_TLM6c_multitemp_mylabels_val.csv'.format(main_input_source, 
                                                                                    aux_input_source)
new_history = True
starting_model_name = 'Unet_SI2020_100cm_50lumagrayaugment_large_lrsched'
starting_model_fn =  os.path.join('output', 
                                    starting_model_name, 
                                    'training', 
                                    '{}_model.pt'.format(starting_model_name))
resume_training = True #  False # must be True to train GRUUnet
freeze_matching_params = 0 #num_epochs
model_arch = 'GRUUnet' #'NonRecurrentUnet' #
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
batch_size = 8 # 64 for frozen U-net
update_period = 256 // batch_size # simulate a larger batch size. 
bn_momentum = 1e-5 #0.001 for batch_size 64

# data augmentation
augment_vals = True 
gauss_blur_sigma = 0.25
color_jitter = 0.25
grayscale_prob = 0
std_gray_noise = 0

# loss
lambda_temp = 1.0 #20
temp_loss = 'CE' #MSE' #'expansion' # , 'none' #'graddot'
lambda_temp_align = 1.0
temp_align_loss = 'gradnorm' #'graddot'
scale_by_norm = True
asym_align = True
weight_temp_loss = True
bootstrap_beta = 0.0 # 0.5
bootstrap_threshold = 1.0 #0.5 #1 #0.75

# temporal model
reverse = True
gru_irreg = True
gru_kernel_size = 7
gru_input = 'df'
gru_norm_dt = True and gru_irreg

# data pre-processing
common_input_bands = None

# resource allocation
num_workers_train = 8
num_workers_val = 4

# misc
no_user_input = True
debug=False


# for lambda_temp in [20, 10]:
#     for freeze_matching_params in [0, 5, num_epochs]:
# for gru_irreg in [True, False]:
# for bootstrap_beta, bootstrap_threshold in [(0., 1.), (0.25, 0.75)]:
# for lr_fe, lr_temp, lambda_temp, temp_loss, lambda_temp_align, temp_align_loss, random_seed in [
#                                                             (1e-6, 1e-4, 2, 'CE', 0., 'none', 2), 
#                                                             (1e-6, 1e-4, 20, 'MSE', 0., 'none', 2),
#                                                             (1e-6, 1e-4, 1., 'CE', 1., 'graddot', 2), 
#                                                             (1e-6, 1e-4, 10, 'MSE', 1., 'graddot', 2),
#                                                             ]: 

# for temp_loss, temp_align_loss, lambda_temp, lambda_temp_align, asym_align in [
#     ('CE', 'graddot', 1.0, 1.0, True),
#     ('CE', 'graddot', 1.0, 1.0, False),
#     ('CE', 'none', 2.0, 0.0, False),
#     ('MSE', 'graddot', 1.0, 1.0, False)]:
    
for random_seed in range(5):
    if model_arch == 'GRUUnet':
        if gru_irreg:
            model_arch_name = 'NIrregGRU' if gru_norm_dt else 'IrregGRU'
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

    exp_name = '{}{}{}_{}_freeze{}_lrfe{}_lrtemp{}_tloss{}_ltemp{}_rs{}'.format(
                                model_arch_name,
                                gru_kernel_size,
                                gru_input,
                                'bwrd' if reverse else 'fwrd',
                                freeze_matching_params,
                                lr_fe,
                                lr_temp,
                                temp_loss_name,
                                lambda_temp_name,
                                random_seed)
    exp_name = exp_name.replace('.', '_')
    exp_name = exp_name.replace('-', 'm')
    # exp_name = 'debug'
    output_dir = os.path.join('output', exp_name)
    
    # if os.path.exists(output_dir):
    #     print('{} exists. Skipping it'.format(output_dir))
    
    # else:
    if True:

        print('Launching experiment {}'.format(exp_name))

        train(  output_dir=output_dir,
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
                            std_gray_noise=std_gray_noise,
                            lambda_temp=lambda_temp,
                            temp_loss=temp_loss,
                            lambda_temp_align=lambda_temp_align,
                            temp_align_loss=temp_align_loss,
                            scale_by_norm=scale_by_norm,
                            asym_align=asym_align,
                            weight_temp_loss=weight_temp_loss,
                            bootstrap_beta=bootstrap_beta,
                            bootstrap_threshold=bootstrap_threshold,
                            reverse=reverse,
                            gru_irreg=gru_irreg,
                            gru_kernel_size=gru_kernel_size,
                            gru_input=gru_input,
                            gru_norm_dt=gru_norm_dt,
                            common_input_bands=common_input_bands,
                            num_workers_train=num_workers_train,
                            num_workers_val=num_workers_val,
                            debug=debug,
                            no_user_input=no_user_input,
                            wandb_name=exp_name,
                            wandb_tracking=wandb_tracking)

