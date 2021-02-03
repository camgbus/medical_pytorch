import os
import argparse
import traceback
from mp.paths import storage_data_path
from train_restore_use_models.UNET_train import UNet2D_initialize_and_train


parser = argparse.ArgumentParser(description='Train a specified model for augmented CT scans.')
parser.add_argument('--device', action='store', type=int, nargs=1, default=4,
                    help='Try to train the model on the GPU device with <DEVICE> ID.'+
                         ' Valid IDs: 0, 1, ..., 7'+
                         ' Default: GPU device with ID 4 will be used.')
parser.add_argument('--resize', choices=['True','False'], required=True)

args = parser.parse_args()
device = args.device
resize = bool(args.resize)

config = {'device':'cuda:{}'.format(device[0]), 'nr_runs': 1, 'cross_validation': False, 
          'val_ratio': 0.2, 'test_ratio': 0.2, 'input_shape': (1, 256,256),
          'resize': resize, 'augmentation': 'none', 'lr': 0.001, 'batch_size':64,
          'nr_epochs': 30,  'save_interval': 1,
          'augmented': True, 'dataset': 'UKF2',
          'class_weights': (0.,1.)
         }

UNet2D_initialize_and_train(config) 