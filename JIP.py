import os
import time
import argparse
from mp.paths import JIP_dir, telegram_login
from mp.utils.update_bots.telegram_bot import TelegramBot
from train_restore_use_models.preprocess_data import preprocess_data
from train_restore_use_models.CNN_train_restore_use import train_model
from train_restore_use_models.CNN_train_restore_use import restore_train_model
from train_restore_use_models.CNN_train_restore_use import retrain_model
from train_restore_use_models.CNN_train_restore_use import test_model
from train_restore_use_models.CNN_train_restore_use import do_inference

# Structure of JIP_dir/data_dirs:
# /
# |---WORKFLOW_DIR --> JIP_dir/data_dirs for inference or JIP_dir/train_dirs for training or JIP_dir/preprocessed_dirs for preprocessed data
#     |---OPERATOR_IN_DIR
#     |   |---0001
#     |   |   |---img
#     |   |   |   |---img.nii
#     |   |   |---seg
#     |   |       |---001.nii
#     |   |       |---002.nii
#     |   |       |---...
#     |   |---0002
#     |   |   |---img
#     |   |   |   |---img.nii
#     |   |   |---seg
#     |   |       |---001.nii
#     |   |       |---002.nii
#     |   |       |---...
#     |   |---...
#     |---OPERATOR_OUT_DIR
#     |---OPERATOR_TEMP_DIR

if __name__ == "__main__": 
    # Build Argumentparser
    parser = argparse.ArgumentParser(description='Train, reterain or use a specified model to predict the quality of CT scans.')
    parser.add_argument('--noise_type', choices=['blur', 'resolution', 'ghosting', 'noise',
                                                'motion', 'spike'], required=False,
                        help='Specify the CT artefact on which the model will be trained. '+
                             'Default model type: blur.')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'retrain', 'testID', 'testOOD', 'testIOOD', 'inference'], required=True,
                        help='Specify in which mode to use the model. Either train a model or use'+
                             ' it for predictions. This can also be used to preprocess data (be)for(e) training.')
    parser.add_argument('--datatype', choices=['all', 'train', 'test', 'inference'], required=False,
                        help='Only necessary for mode preprocessing. Indicates which data should be preprocessed.'+
                             ' If not specified, \'all\' will be used for preprocessing.')
    parser.add_argument('--device', action='store', type=int, nargs=1, default=4,
                        help='Try to train the model on the GPU device with <DEVICE> ID.'+
                            ' Valid IDs: 0, 1, ..., 7'+
                            ' Default: GPU device with ID 4 will be used.')
    parser.add_argument('--restore', action='store_const', const=True, default=False,
                        help='Restore last saved model state and continue training from there.'+
                             ' Restore is also used in the augmentation step of the preprocessing of the train_ds only,'+
                             ' if the augmentation fails at some point. --> This flag is ambigious!'
                            ' Default value: False.')
    parser.add_argument('--use_telegram_bot', action='store_const', const=True, default=False,
                        help='Send message during training through a Telegram Bot'+
                            ' (Token and Chat-ID need to be provided, otherwise an error occurs!).'+
                            ' Default: No Telegram Bot will be used to send messages.')
    parser.add_argument('--testID', action='store_const', const=True, default=False,
                        help='Test the pre-trained model with an unseen In Distribution dataset.'+
                            ' (Only considered if --mode test is used).'+
                            ' Default: Used if not otherwise specified.')
    parser.add_argument('--testOOD', action='store_const', const=True, default=False,
                        help='Test the pre-trained model with an unseen Out Of Distribution dataset.'+
                            ' (Only considered if --mode test is used).'+
                            ' Default: If not set, --testID will be used.')
    parser.add_argument('--testIOOD', action='store_const', const=True, default=False,
                        help='Test the pre-trained model with an unseen In Distribution'+
                            ' and an Out Of Distribution dataset.'+
                            ' (Only considered if --mode test is used).'+
                            ' Default: If not set, --testID will be used.')
    parser.add_argument('--try_catch_repeat', action='store', type=int, nargs=1, default=0,
                        help='Try to train the model with a restored state, if an error occurs.'+
                            ' Repeat only <TRY_CATCH_REPEAT> number of times. It can also be used for'+
                            ' preprocessing step, since the augmentation might fail at some point, for'+
                            ' training on a model and retraining a model.'+
                            ' Default value: 0.')
    parser.add_argument('--idle_time', action='store', type=int, nargs=1, default=0,
                        help='The number represents the idle time (waiting time) in seconds after an error'+
                            ' occurs before starting the process again. --> Works only in combination'+
                            ' with --try_catch_repeat.'+
                            ' Default: Idle time is 0 seconds.')

    # Define configuration dict and train the model
    args = parser.parse_args()
    noise = args.noise_type
    mode = args.mode
    data_type = args.datatype
    cuda = args.device
    restore = args.restore
    msg_bot = args.use_telegram_bot
    try_catch = args.try_catch_repeat
    idle_time = args.idle_time

    if isinstance(cuda, list):
        cuda = cuda[0]
    if isinstance(try_catch, list):
        try_catch = try_catch[0]
    if isinstance(idle_time, list):
        idle_time = idle_time[0]

    if mode == 'preprocess' and data_type is None:
        data_type = 'all'
    if mode != 'preprocess' and noise is None:
        noise = 'blur'
   
    # Define Telegram Bot
    if msg_bot:
        bot = TelegramBot(telegram_login)

    if cuda < 0 or cuda > 7:
        bot.send_msg('GPU device ID out of range (0, ..., 7).')
        assert False, 'GPU device ID out of range (0, ..., 7).'
    cuda = 'cuda:' + str(cuda)

    # -------------------------
    # Build environmental vars
    # -------------------------
    print('Building environmental variables..')
    # The environmental vars will later be automatically set by the workflow that triggers the docker container
    # data_dirs (for inference)
    os.environ["WORKFLOW_DIR"] = os.path.join(JIP_dir, 'data_dirs')
    os.environ["OPERATOR_IN_DIR"] = "input"
    os.environ["OPERATOR_OUT_DIR"] = "output"
    os.environ["OPERATOR_TEMP_DIR"] = "temp"
    os.environ["OPERATOR_PERSISTENT_DIR"] = os.path.join(JIP_dir, 'data_dirs', 'persistent') # pre-trained models

    # preprocessed_dirs (for preprocessed data (output of this workflow = input for main workflow)
    os.environ["PREPROCESSED_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'preprocessed_dirs')
    os.environ["PREPROCESSED_OPERATOR_OUT_TRAIN_DIR"] = "output_train"
    os.environ["PREPROCESSED_OPERATOR_OUT_TEST_DIR"] = "output_test"
    os.environ["PREPROCESSED_OPERATOR_OUT_DATA_DIR"] = "output_data"

    # train_dirs (for training data)
    os.environ["TRAIN_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'train_dirs')

    # test_dirs (for test data)
    os.environ["TEST_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'test_dirs')

    # ------------------------
    # Build config dictionary
    # ------------------------
    # NOTE: Decathlon Dataset will be nr_images x 5 scans big.
    #       Grand Challenge Dataset will be nr_images x 5 scans big (or less if less scans are available).
    #       UK Frankfurt Dataset will be nr_images x 5 scans big (or less if less scans are available).
    # NOTE: num_intensities embodies the number of quality values 1 to 5), where 1 is a bad quality
    #       and 5 is a the best quality. This will be transformed into values between 0 and 1 in
    #       the inference step, whereas 0s is bad quality and 1 is the best quality.
    # NOTE: train_on is includes all datasets, on which the model needs to be trained. The list will be used for EWC
    #       approach. For the conventional approach, only the first entry of the list will be considered.
    #       {'Decathlon', 'GC', 'FRA', 'mixed'}. For institutes that use this method to retrain a pre-trained model
    #       with their own dataset can write every name they want, since the whole data from preprocessed_dirs/output_train
    #       will be loaded and used. Thus, this variable will not be considered in this process.
    #       Its important to note, that the labels should be also provided at preprocessed_dirs/output_train/labels.json
    #       and that the data in preprocessed_dirs/output_train needs to be preprocessed while considering the defined data structure.
    config = {'device': cuda, 'input_shape': (1, 60, 299, 299), 'augmentation': True, 'mode': mode,
              'data_type': data_type, 'lr': 0.001, 'batch_size': 64, 'num_intensities': 5, 'nr_epochs': 150,
              'noise': noise, 'weight_decay': 0.01, 'save_interval': 20, 'msg_bot': msg_bot,
              'bot_msg_interval': 10, 'nr_images': 20, 'val_ratio': 0.2, 'test_ratio': 0.2, 'augment_strat': 'none',
              'train_on': 'mixed', 'data_augmented': True, 'restore': restore}
    
    # -------------------------
    # Preprocess
    # -------------------------
    if mode == 'preprocess':
        dir_name = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_TRAIN_DIR"])
        augmented_on = os.path.join(dir_name, 'augmented_on.pkl')
        if try_catch == 0:
            try_catch = 1
        for i in range(try_catch):
            if i != 0:
                time.sleep(idle_time)   # Delays for idle_time seconds.
            if not restore:
                if msg_bot:
                    bot.send_msg('Start to preprocess data..')
                preprocessed, error = preprocess_data(config)
                if preprocessed and msg_bot:
                    bot.send_msg('Finished preprocessing..')
                if not preprocessed:
                    print('Data could not be preprocessed. The following error occured: {}.'.format(error))
                    if msg_bot:
                        bot.send_msg('Data could not be preprocessed. The following error occured: {}.'.format(error))
                    # Check if the directory is empty and whether the augmented_on pickle file does not exist. If so, restore = False, otherwise True.
                    if (os.path.exists(dir_name) and os.path.isdir(dir_name)):
                        if len(os.listdir(dir_name)) <= 1 or not os.path.exists(augmented_on):
                            # Directory contains nothing
                            restore = False
                            config['restore'] = False
                        else:
                            # Directory is not empty
                            restore = True
                            config['restore'] = True
            else:
                if msg_bot:
                    bot.send_msg('Start to continue with preprocessing..')
                preprocessed, error = preprocess_data(config)
                if preprocessed and msg_bot:
                    bot.send_msg('Finished preprocessing..')
                    break
                if not preprocessed:
                    print('Data could not be preprocessed. The following error occured: {}.'.format(error))
                    if msg_bot:
                        bot.send_msg('Data could not be preprocessed. The following error occured: {}.'.format(error))
    

    # -------------------------
    # Train
    # -------------------------
    if mode == 'train':
        dir_name = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], noise, 'states')
        if try_catch == 0:
            try_catch = 1
        for i in range(try_catch):
            if i != 0:
                time.sleep(idle_time)   # Delays for idle_time seconds.
            if not restore:
                if msg_bot:
                    bot.send_msg('Start to initialize and train the model for noise type {}..'.format(noise))
                trained, error = train_model(config)
                if trained and msg_bot:
                    bot.send_msg('Finished training for noise type {}..'.format(noise))
                    break
                if not trained:
                    print('Model for noise type {} could not be initialized/trained. The following error occured: {}.'.format(noise, error))
                    if msg_bot:
                        bot.send_msg('Model for noise type {} could not be trained. The following error occured: {}.'.format(noise, error))
                    # Only restore, if a model state has already been saved, otherwise Index Error
                    # occurs while trying to extract the highest saved state for restoring a state.
                    # Check if the directory is empty. If so, restore = False, otherwise True.
                    if os.path.exists(dir_name) and os.path.isdir(dir_name):
                        if len(os.listdir(dir_name)) <= 1:
                            # Directory only contains json splitting file but no model state!
                            restore = False
                            config['restore'] = False
                        else:
                            # Directory is not empty
                            restore = True
                            config['restore'] = True
            else:
                if msg_bot:
                    bot.send_msg('Start to restore the model for noise type {} and continue training..'.format(noise))
                trained, error = restore_train_model(config)
                if trained and msg_bot:
                    bot.send_msg('Finished training for noise type {}..'.format(noise))
                    break
                if not trained:
                    print('Model for noise type {} could not be restored/trained. The following error occured: {}.'.format(noise, error))
                    if msg_bot:
                        bot.send_msg('Model for noise type {} could not be restored/trained. The following error occured: {}.'.format(noise, error))
    
    # -------------------------
    # Retrain
    # -------------------------
    if mode == 'retrain':
        dir_name = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], noise, 'states')
        if try_catch == 0:
            try_catch = 1
        for i in range(try_catch):
            if i != 0:
                time.sleep(idle_time)   # Delays for idle_time seconds.
            if not restore:
                if msg_bot:
                    bot.send_msg('Start to retrain the model for noise type {}..'.format(noise))
                trained, error = retrain_model(config)
                if trained and msg_bot:
                    bot.send_msg('Finished retraining for noise type {}..'.format(noise))
                    break
                if not trained:
                    print('Model for noise type {} could not be retrained. The following error occured: {}.'.format(noise, error))
                    if msg_bot:
                        bot.send_msg('Model for noise type {} could not be retrained. The following error occured: {}.'.format(noise, error))
                    # Only restore, if a model state has already been saved, otherwise Index Error
                    # occurs while trying to extract the highest saved state for restoring a state.
                    # Check if the directory is empty. If so, restore = False, otherwise True.
                    if os.path.exists(dir_name) and os.path.isdir(dir_name):
                        if len(os.listdir(dir_name)) <= 1:
                            # Directory only contains json splitting file but no model state!
                            restore = False
                            config['restore'] = False
                        else:
                            # Directory is not empty
                            restore = True
                            config['restore'] = True
            else:
                if msg_bot:
                    bot.send_msg('Start to restore the model for noise type {} and continue training..'.format(noise))
                trained, error = restore_train_model(config)
                if trained and msg_bot:
                    bot.send_msg('Finished retraining for noise type {}..'.format(noise))
                    break
                if not trained:
                    print('Model for noise type {} could not be restored/retrained. The following error occured: {}.'.format(noise, error))
                    if msg_bot:
                        bot.send_msg('Model for noise type {} could not be restored/retrained. The following error occured: {}.'.format(noise, error))

    # -------------------------
    # Test
    # -------------------------
    if 'test' in mode:
        if msg_bot:
            bot.send_msg('Start testing the pre-trained model..')
        tested, error = test_model(config)
        if tested and msg_bot:
            bot.send_msg('Finished testing..')
        if not tested:
            print('Testing could not be performed. The following error occured: {}.'.format(error))
            if msg_bot:
                bot.send_msg('Testing could not be performed. The following error occured: {}.'.format(error))

    # -------------------------
    # Inference
    # -------------------------
    if mode == 'inference':
        if msg_bot:
            bot.send_msg('Start the inference..')
        inferred, error = do_inference(config)
        if inferred and msg_bot:
            bot.send_msg('Finished inference..')
        if not inferred:
            print('Inference could not be performed. The following error occured: {}.'.format(error))
            if msg_bot:
                bot.send_msg('Inference could not be performed. The following error occured: {}.'.format(error))