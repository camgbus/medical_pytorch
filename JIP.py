import sys
import os
import argparse
from mp.paths import JIP_dir, telegram_login
from mp.utils.update_bots.telegram_bot import TelegramBot
from train_restore_use_models.preprocess_data import preprocess_data
from train_restore_use_models.CNN_train_restore_use import train_model
from train_restore_use_models.CNN_train_restore_use import restore_train_model
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
    parser.add_argument('--noise_type', choices=['blur', 'downsample', 'ghosting', 'noise',
                                                'motion', 'spike'], required=False,
                        help='Specify the CT artefact on which the model will be trained. '+
                             'Default model type: blur.')
    #parser.add_argument('--model_type', choices=['cnn'], required=False,
    #                    help='Specify the model type that will be trained.')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'inference'], required=True,
                        help='Specify in which mode to use the model. Either train a model or use'+
                             ' it for predictions. This can also be used to preprocess data (be)for(e) training.')
    parser.add_argument('--datatype', choices=['all', 'train', 'inference'], required=False,
                        help='Only necessary for mode preprocessing. Indicates which data should be preprocessed.'+
                             ' If not specified, \'all\' will be used for preprocessing.')
    parser.add_argument('--device', action='store', type=int, nargs=1, default=4,
                        help='Try to train the model on the GPU device with <DEVICE> ID.'+
                            ' Valid IDs: 0, 1, ..., 7'+
                            ' Default: GPU device with ID 4 will be used.')
    parser.add_argument('--restore', action='store_const', const=True, default=False,
                        help='Restore last saved model state and continue training from there.'+
                            ' Default: Initialize a new model and train from beginning.')
    parser.add_argument('--use_telegram_bot', action='store_const', const=True, default=False,
                        help='Send message during training through a Telegram Bot'+
                            ' (Token and Chat-ID need to be provided, otherwise an error occurs!).'+
                            ' Default: No Telegram Bot will be used to send messages.')
    parser.add_argument('--try_catch_repeat', action='store', type=int, nargs=1, default=0,
                        help='Try to train the model with a restored state, if an error occurs.'+
                            ' Repeat only <TRY_CATCH_REPEAT> number of times.'+
                            ' Default: Do not retry to train after an error occurs.')

    # 5. Define configuration dict and train the model
    args = parser.parse_args()
    noise = args.noise_type
    #model = args.model_type
    mode = args.mode
    data_type = args.datatype
    cuda = args.device
    restore = args.restore
    msg_bot = args.use_telegram_bot
    try_catch = args.try_catch_repeat
    if isinstance(cuda, list):
        cuda = cuda[0]
    if isinstance(try_catch, list):
        try_catch = try_catch[0]

    if mode == 'preprocess' and data_type is None:
        data_type = 'all'
    
    if mode != 'preprocess' and noise is None:
        noise = 'blur'

        
    # 6. Define Telegram Bot
    if msg_bot:
        bot = TelegramBot(telegram_login)

    if cuda < 0 or cuda > 7:
        bot.send_msg('GPU device ID out of range (0, ..., 7).')
        assert False, 'GPU device ID out of range (0, ..., 7).'

    cuda = 'cuda:' + str(cuda)

    # -------------------------
    # Build environmental vars
    # -------------------------
    print('Building environmental variables')
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
    os.environ["PREPROCESSED_OPERATOR_OUT_DATA_DIR"] = "output_data"

    # train_dirs (for training data)
    os.environ["TRAIN_WORKFLOW_DIR"] = os.path.join(JIP_dir, 'train_dirs')
    
    # In order to use these directories, always join them to WORKFLOW_DIR
    # Only exception is the OPERATOR_PERSISTENT_DIR since it probably won't be located inside the WORKFLOW_DIR
    
    """
    # Example how all dirs should be used:
    input_dir = os.path.join(os.environ["WORKFLOW_DIR"], os.environ["OPERATOR_IN_DIR"])
    output_dir = os.path.join(os.environ["WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"])
    temp_dir = os.path.join(os.environ["WORKFLOW_DIR"], os.environ["OPERATOR_TEMP_DIR"])
    persistent_dir = os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"])
    """

    # ------------------------
    # Build config dictionary
    # ------------------------
    # Note: Dataset will be nr_images x 5 big!
    # Note: num_intensities embodies the number of quality values 1 to 5), where 1 is a bad quality
    #       and 5 is a the best quality. This will be transformed into values between 0 and 1 in
    #       the inference step, whereas 0s is bad quality and 1 is the best quality.
    config = {'device':cuda, 'input_shape':(1, 60, 299, 299), 'augmentation': True,
              'data_type': data_type, 'lr': 0.001, 'batch_size': 64, 'num_intensities': 5, 'nr_epochs': 100,
              'noise': noise, 'weight_decay': 0.75, 'save_interval': 25, 'msg_bot': msg_bot,
              'bot_msg_interval': 10, 'nr_images': 20, 'val_ratio': 0.2, 'test_ratio': 0.2, 'augment_strat': 'none'}


    # -------------------------
    # Preprocess
    # -------------------------
    if mode == 'preprocess':
        if msg_bot:
            bot.send_msg('Start to preprocess data..')
        preprocessed, error = preprocess_data(config)
        if preprocessed and msg_bot:
            bot.send_msg('Finished preprocessing..')
        if not preprocessed:
            print('Data could not be processed. The following error occured: {}.'.format(error))
            if msg_bot:
                bot.send_msg('Data could not be processed. The following error occured: {}.'.format(error))

    # -------------------------
    # Train
    # -------------------------
    if mode == 'train':
        dir_name = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], noise, 'states')
        if try_catch == 0:
            try_catch = 1
        for i in range(try_catch):
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
                        else:
                            # Directory is not empty
                            restore = True
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
                        
        """ Without try-catch-repeat
        if not restore:
            if msg_bot:
                bot.send_msg('Start to initialize and train the model for noise type {}..'.format(noise))
            trained, error = train_model(config)
            if trained and msg_bot:
                bot.send_msg('Finished training for noise type {}..'.format(noise))
            if not trained:
                print('Model for noise type {} could not be initialized/trained. The following error occured: {}.'.format(noise, error))
                if msg_bot:
                    bot.send_msg('Model for noise type {} could not be trained. The following error occured: {}.'.format(noise, error))
        else:
            if msg_bot:
                bot.send_msg('Start to restore the model for noise type {} and continue training..'.format(noise))
            trained, error = restore_train_model(config)
            if trained and msg_bot:
                bot.send_msg('Finished training for noise type {}..'.format(noise))
            if not trained:
                print('Model for noise type {} could not be restored/trained. The following error occured: {}.'.format(noise, error))
                if msg_bot:
                    bot.send_msg('Model for noise type {} could not be restored/trained. The following error occured: {}.'.format(noise, error))
        """

    # -------------------------
    # Interference
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


"""
# -------------------------------------------------------------------------------------------------------

# The following shows the usage of the DataConnector to obtain images / segmentations
# from these environ paths. It also provides a method for aggregating the metrics and writing
# them as a JSON file to the output path.

# You don't actually have to use my DataConnector. It's also okay if you just pass the paths to 
# your own classes. There are only two points you should adhere:
# 1. Do not use directories outside the specified ones from above
# 2. Make sure that after completion there is a file called "metrics.json" inside the OPERATOR_OUT_DIR

# -------------------------------------------------------------------------------------------------------

# -------------------------
# Initialize Data Connector
# -------------------------

dc = DataConnector(extension="mhd")
dc.loadData()    

# ---------------------------
# --- DataConnector Usage ---
# ---------------------------

# Obtain all segmentation and image paths:    
seg_path_list = dc.getAllSeg()
img_path_list = dc.getAllImg()

# Obtain all segmentations and images loaded as ITK:    
seg_itk_list = dc.getAllSegAsItk()
img_itk_list = dc.getAllImgAsItk()

# Obtain data paths pairwise:
for inst in dc.instances:
    seg_path = inst.seg
    img_path = inst.img

# Obtain data as ITK pairwise:
for inst in dc.instances:
    seg_itk = inst.getItkSeg()
    img_itk = inst.getItkImg()

# ==================================
# --- ADD MEASUREMENT CALCS HERE ---
# ==================================

# Init Quantifier
exampleQuantifier = ExampleQuantifier(version="1.0")

# Get Data from DataConnector
segmentations = dc.getAllSeg()
imgs = dc.getAllImg()
    
# Calc Metrics:
metrics = exampleQuantifier.get_quality(x=imgs, mask=segmentations)

# Append Metrics:
dc.appendMetric(metrics)
dc.appendMetric({'m01': 68.697, 'm02': 'test'})    

# -------------------
# Create Final Output
# -------------------

# write all metrics as JSON to the workspace output dir
dc.createOutputJson()   
"""