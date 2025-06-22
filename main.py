import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import numpy
import torch


def main(config):
    # reproducibility
    seed = 3047
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    decay_ratio = 0.1
    decay_epoch = int(config.num_epochs * decay_ratio)
    config.model_path = os.path.join(config.model_path,
                                     f'models_lr_{config.lr}_bs_{config.batch_size}')
    config.result_path = os.path.join(config.result_path,
                                      f'results_lr_{config.lr}_bs_{config.batch_size}')
    config.num_epochs_decay = decay_epoch

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    print(config)

    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob,
                              image_type=config.image_type,
                              exp_or_sim=config.exp_or_sim,
                              config=config)
    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=16,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0.,
                              exp_or_sim=config.exp_or_sim,
                              image_type=config.image_type,
                              config=config)
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.,
                             exp_or_sim=config.exp_or_sim,
                             image_type=config.image_type,
                             config=config)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test(pretrain_path=config.test_pretrained_model_path)
    elif config.mode == 'generate':
        solver.generate_test_result()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    image_type = '8bits'  # '8bits' or '16bits'
    dir_name = os.path.dirname(os.path.abspath(__file__))

    # ---------------------------> Parameter group <--------------------------- #
    NUM_L = 10
    exp_or_sim = 'exp'
    train_valid_folder = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', ]
    # these are the folders for the partitions
    encoder_only = True
    output_hw = 64
    record_train_valid_name = f'enc_only_{encoder_only}_output_hw_{output_hw}'
    for f_name in train_valid_folder:
        record_train_valid_name += (f_name + '_')
    # ========================================================================= #

    # for training on exp data
    parser.add_argument('--selected_train_valid_fold', type=list, default=train_valid_folder)

    parser.add_argument('--special_save_folder_name', type=str,
                        default=f'{NUM_L}L_{exp_or_sim}_transformerUnet_folders_{record_train_valid_name}',
                        help='the model would be saved as xxx_{special_save_name}.pkl')
    parser.add_argument('--special_save_name', type=str,
                        default=f'{NUM_L}L_{exp_or_sim}_transformerUnet_BS64_folders_{record_train_valid_name}',
                        help='the model would be saved as xxx_{special_save_name}.pkl')

    parser.add_argument('--mode', type=str, default='train', help='train | test | generate')

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--image_type', type=str, default=image_type)

    parser.add_argument('--t', type=int, default=3,
                        help='t for Recurrent step of R2U_Net or R2AttU_Net')
    parser.add_argument('--valid_rate', type=float, default=0.1,
                        help="how much ratio of data in the training set would be set as valid data.")

    # paths
    parser.add_argument('--dir_path', type=str,
                        default=f'/raid/crp.dssi/volume_Kubernetes/Benquan/Dataset_QRcode_Experiment/Dataset_QRcode_180nmStep/10L',
                        help='the path of the dataset')
    parser.add_argument('--save_path', type=str, default=f'save',
                        help='the path to save the model and results')

    # a default pretrained model path
    pretrained_path = '/raid/crp.dssi/volume_Kubernetes/Benquan/data_L/AI_Optics_result_10L/models_lr_0.0001_bs_6410L_exp_transformerUnet_folders_enc_only_True_output_hw_64A1_A2_A3_A4_A5_A6_/Transformer_UNet_epoch_125_lr_0.0001_focus_weight_0.0_8bits_10L_exp_transformerUnet_BS64_folders_enc_only_True_output_hw_64A1_A2_A3_A4_A5_A6_.pkl'

    parser.add_argument('--test_pretrained_model_path', default=pretrained_path, type=str, )

    parser.add_argument('--selected_test_fold', type=list, default=['SS', 'ORC'])

    # training hyper-parameters&
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_epochs_decay', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--wd', type=float, default=1e-5)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)
    parser.add_argument('--focus_weight', type=float, default=0.,
                        help="new_loss = bce_loss + focus_weight * focus_loss")
    parser.add_argument('--focus_beta', type=float, default=0.5, help='beta for f_beta loss')
    parser.add_argument('--model_type', type=str, default='Transformer_UNet',
                        choices=['UNet', 'CustomResNet34', 'CustomResNet18', 'Transformer_UNet'], help='UNet')
    parser.add_argument('--exp_or_sim', type=str, default=None)
    parser.add_argument('--rotate', type=bool, default=True)
    parser.add_argument('--center_crop', type=bool, default=False)
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--cuda_idx', type=int, default=0)

    # for transformer configs
    parser.add_argument('--L', type=int, default=NUM_L, choices=[10],
                        help='L')
    parser.add_argument('--transformer_dim', type=int, default=64,
                        help='channels of the network first layer output')
    parser.add_argument('--encoder_only', type=bool, default=encoder_only,
                        choices=[True, False],
                        help='whether use the decoder to predict a whole image')
    parser.add_argument('--output_hw', type=int, default=output_hw,
                        help='linear output shape = hw * hw')

    config = parser.parse_args()

    parser.train_path = os.path.join(config.dir_path, 'train_valid')
    parser.valid_path = os.path.join(config.dir_path, 'train_valid')
    parser.test_path = os.path.join(config.dir_path, 'test')

    parser.model_path = config.save_path
    parser.result_path = config.save_path

    # wandb.login()
    # wandb.init(project='QR_code_experiment_180nmStep_transformer_Unet', entity='ntubenquan', name=config.special_save_name, tags=['baseline'])
    # wandb.config.update(vars(config))

    if config.mode == 'train':
        try:
            main(config)
        except Exception as e:
            raise e
    elif config.mode == 'test':
        test_fold = config.selected_test_fold
        config.batch_size = 1

        for fold in test_fold:
            config.selected_test_fold = [fold]
            config.result_path = config.save_path
            main(config)
