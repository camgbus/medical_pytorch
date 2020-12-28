# 1. Import needed libraries and functions
import os
import argparse
import traceback
from mp.paths import storage_data_path, telegram_login
from mp.utils.update_bots.telegram_bot import TelegramBot
from train_restore_use_models.CNN_train_restore_use import CNN_initialize_and_train, CNN_restore_and_train, CNN_predict
from train_restore_use_models.Reg_train_restore_use import Reg_initialize_and_train, Reg_restore_and_train, Reg_predict 

# 2. Train the model based on command line arguments
def train(model, restore, config):
    r"""Trains a model based on the type, config and restore values."""
    if model == 'cnn':
        if restore:
            CNN_restore_and_train(config)
        else:
            CNN_initialize_and_train(config)

    if model == 'regression' or model == 'reg':
        if restore:
            Reg_restore_and_train(config)
        else:
            Reg_initialize_and_train(config)

# 3. Use the model based on command line arguments for predictions
def predict(model, config):
    r"""Predicts values based on the type and config values."""
    if model == 'cnn':
        CNN_predict(config)

    if model == 'regression' or model == 'reg':
        Reg_predict(config)

parser = argparse.ArgumentParser(description='Train a specified model for augmented CT scans.')
parser.add_argument('--noise_type', choices=['blur', 'downsample', 'ghosting', 'noise',
                                             'motion', 'spike'], required=True,
                    help='Specify the CT artefact on which the model will be trained.'+
                         ' Default: The model will be trained on blurred CT images.')
parser.add_argument('--model_type', choices=['cnn', 'regression', 'reg'], required=True,
                    help='Specify the model type that will be trained.'+
                         ' Default: A regression model will be trained.')
parser.add_argument('--mode', choices=['train', 'use'], required=True,
                    help='Specify in which mode to use the model. Either train a model or use'+
                         ' it for predictions.'+
                         ' Default: The model will be trained.')
parser.add_argument('--restore', action='store_const', const=True, default=False,
                    help='Restore last model state and continue training from there.'+
                         ' Default: Initialize a new model and train from beginning.')
parser.add_argument('--use_telegram_bot', action='store_const', const=True, default=False,
                    help='Send message during training through a Telegram Bot'+
                         ' (Token and Chat-ID need to be provided, otherwise an error occurs!).'+
                         ' Default: No Telegram Bot will be used to send messages.')
parser.add_argument('--try_catch_repeat', action='store', type=int, nargs=1, default=0,
                    help='Try to train the model with a restored state, if an error occurs.'+
                         ' Repeat only <TRY_CATCH_REPEAT> number of times.'+
                         ' Default: Do not retry to train after an error occurs.')


"""parser.add_argument('--try_catch_repeat', action='store', type=int, nargs=1, default=0,
                    help='Try to train the model with a restored state, if an error occurs.'+
                         ' Repeat only <TRY_CATCH_REPEAT> number of times. For debugging errors,'+
                         ' this should not be set, since only the error name will be displayed!'+
                         ' Default: Do not retry to train after an error occurs.')"""


# 4. Define configuration dict and train the model
args = parser.parse_args()
noise = args.noise_type
model = args.model_type
mode = args.mode
restore = args.restore
msg_bot = args.use_telegram_bot
try_catch = args.try_catch_repeat
if isinstance(try_catch, list):
    try_catch = try_catch[0]
config = {'device':'cuda:4', 'nr_runs': 1, 'cross_validation': False, 
          'val_ratio': 0.2, 'test_ratio': 0.2, 'input_shape': (1, 299, 299),
          'resize': False, 'augmentation': 'none', 'lr': 0.001, 'batch_size': 64,
          'max_likert_value': 5, 'nr_epochs': 300, 'noise': noise, 
          'random_slices': True, 'nr_images': 40, 'nr_slices': 25,
          'weight_decay': 0.75, 'save_interval': 25, 'msg_bot': msg_bot,
          'bot_msg_interval': 20, 'augmented': True
         }

# 5. Define Telegram Bot
if msg_bot:
    bot = TelegramBot(telegram_login)

if mode == 'train':
    # 6. Train the model until number of epochs is reached. Send every error 
    # with Telegram Bot if desired, however try to repeat training only the
    # transmitted number of times.
    dir_name = os.path.join(storage_data_path, 'models', noise+'_'+model, 'states')
    if try_catch > 0:
        for i in range(try_catch):
            try:
                train(model, restore, config)
                # Break loop if training for number epochs is concluded
                # Otherwise, a permission denied error or other errors occured
                break
            except: # catch *all* exceptions
                e = traceback.format_exc()
                print('Error occured during training {} model for {} noise: {}'.format(model, noise, e))
                if msg_bot:
                    bot.send_msg('Error occured during training {} model for {} noise: {}'.format(model, noise, e))

                # Only restore, if a model state has already been saved, otherwise Index Error
                # occurs while trying to extract the highest saved state for restoring a state.
                # Check if the directory is empty. If so, restore = False, otherwise True.
                if os.path.exists(dir_name) and os.path.isdir(dir_name):
                    if len(os.listdir(dir_name)) <= 1:
                        # Directory only contains json splitting file but no model state!
                        restore = False
                    else:    
                        # Directory is not empty
                        restore = True
                else:
                    # Directory does not exist
                    restore = False

    else:
        train(model, restore, config)
else:
    # 7. Use a pretrained model for predictions. Send every error 
    # with Telegram Bot if desired.
    predict(model, config)