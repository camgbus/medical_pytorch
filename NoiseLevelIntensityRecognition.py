# 1. Import needed libraries and functions
import sys
import argparse
from mp.paths import telegram_login
from mp.utils.update_bots.telegram_bot import TelegramBot
from train_models.CNN_train_and_restore import CNN_initialize_and_train, CNN_restore_and_train
from train_models.Reg_train_and_restore import Reg_initialize_and_train, Reg_restore_and_train 

# 2. Train the model based on terminal arguments
def execute(model, restore, config):
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

parser = argparse.ArgumentParser(description='Train a specified model for augmented CT scans.')
parser.add_argument('--noise_type', choices=['blur', 'downsample', 'ghosting', 'noise',
                                             'motion', 'spike'], required=True,
                    help='Specify the CT artefact on which the model will be trained.'+
                         ' Default: The model will be trained on blurred CT images.')
parser.add_argument('--model_type', choices=['cnn', 'regression', 'reg'], required=True,
                    help='Specify the model type that will be trained.'+
                         ' Default: A regression model will be trained.')
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

# 4. Define configuration dict and train the model
args = parser.parse_args()
noise = args.noise_type
model = args.model_type
restore = args.restore
msg_bot = args.use_telegram_bot
try_catch = args.try_catch_repeat[0]
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

# 6. Train the model until number of epochs is reached. Send every error 
# with Telegram Bot if desired, however repeat only the transmitted number of times.
if try_catch > 0:
    for i in range(try_catch):
        try:
            execute(model, restore, config)
            # Break endless loop if training for number epochs is concluded
            # Otherwise, a permission denied error or other errors occured
            break
        except: # catch *all* exceptions
            e = sys.exc_info()[0]
            print('Error occured during training {} model for {} noise: {}'.format(model, noise, e))
            if msg_bot:
                bot.send_msg('Error occured during training {} model for {} noise: {}'.format(model, noise, e))
            restore = True

else:
    execute(model, restore, config)