# Import needed libraries
import torch
import os
import shutil
import traceback
import random
import SimpleITK as sitk
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.data.data import Data
from mp.data.datasets.dataset_JIP_cnn import JIPDataset
from mp.experiments.data_splitting import split_dataset
import mp.utils.load_restore as lr
from mp.data.pytorch.pytorch_cnn_dataset import PytorchCNN2DDataset, Pytorch3DQueue
from mp.models.cnn.cnn import CNN_Net2D, CNN_Net3D
from mp.eval.losses.losses_cnn import LossCEL, EWC
from mp.agents.cnn_agents import NetAgent
from mp.utils.save_results import save_results, save_only_test_results
from mp.quantifiers.NoiseQualityQuantifier import NoiseQualityQuantifier
from mp.data.DataConnectorJIP import DataConnector


def train_model(config):
    r""" - This function tries to intializes and trains a model.
         - This function tries to intializes and trains a model using the EWC approach.
        It returns True if the model was sucessfully trained and if not an error will be returned as well."""
    try:
        _CNN_initialize_and_train(config)
        #_CNN_initialize_and_train_ewc(config, False)
        return True, None
    except: # catch *all* exceptions
        e = traceback.format_exc()
        return False, e


def restore_train_model(config):
    r""" - This function tries to restore and trains a model. 
         - This function tries to restore and trains a model using the EWC approach.s
        It returns True if the model was sucessfully trained and if not an error will be returned as well."""
    try:
        _CNN_restore_and_train(config)
        #_CNN_restore_and_train_ewc(config)
        return True, None
    except: # catch *all* exceptions
        e = traceback.format_exc()
        return False, e


def retrain_model(config):
    r""" - This function retrains a model (using transfer learning).
         - This function loads a pre-trained model from presistent given the noise
        and retrains this model using the EWC approach --> No transfer learning.
        NOTE: For this approach, all the datasets on which the model has been
        trained on need to be present. It returns True if the model was
        sucessfully retrained and if not an error will be returned as well."""s
    try:
        _CNN_retrain(config)
        #_CNN_initialize_and_train_ewc(config, True)
        return True, None
    except: # catch *all* exceptions
        e = traceback.format_exc()
        return False, e


def do_inference(config):
    r"""This function uses a pretrained model and performs inference on the dataset. It returns True
        if the inference was sucessful and if not an error will be returned as well."""
    try:
        _CNN_predict(config)
        return True, None
    except: # catch *all* exceptions
        e = traceback.format_exc()
        return False, e


# -------------------------
# Standard approach
# -------------------------
def _CNN_initialize_and_train(config):
    r"""This function selects random images etc. based on the config file
        and starts training the model. If everything works fine, without
        and error, the results will be saved."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['num_intensities']
    dataset_name = config['train_on'][0]

    # 2. Define data
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], gpu=True, cuda=config['device'], msg_bot = config['msg_bot'],\
                     nr_images=config['nr_images'], build_dataset=True, dtype='train', noise=config['noise'],\
                     ds_name=dataset_name)

    data.add_dataset(JIP)
    train_ds = (dataset_name, 'train')
    val_ds = (dataset_name, 'val')
    test_ds = (dataset_name, 'test')

    # 3. Split data and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio = config['test_ratio'], 
                          val_ratio = config['val_ratio'], nr_repetitions = 1, cross_validation = False)
    paths = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'states')
    pathr = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'results')
    if not os.path.exists(paths):
        os.makedirs(paths)
    else:
        # Empty directory
        shutil.rmtree(paths)
        os.makedirs(paths)
    if not os.path.exists(pathr):
        os.makedirs(pathr)
    else:
        # Empty directory
        shutil.rmtree(pathr)
        os.makedirs(pathr)

    # Save split
    if splits is not None:
        lr.save_json(splits, path = paths, name = 'data_splits')

    # 4. Create data splits for each repetition
    print('Bring data to PyTorch format..')

    # 5. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in splits[ds_name][0].items():
            if len(data_ixs) > 0: # Sometimes val indices may be an empty list
                aug = config['augment_strat'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = Pytorch3DQueue(ds, 
                    ix_lst = data_ixs, size = (1, 299, 299, 10), aug_key = aug, 
                    samples_per_volume = 10)

    # 6. Build train dataloader
    dl = DataLoader(datasets[(train_ds)], 
        batch_size = config['batch_size'], shuffle = True,
        num_workers = 1)
    dl_val = DataLoader(datasets[(val_ds)], 
        batch_size = config['batch_size'], shuffle = True,
        num_workers = 1)

    # 7. Initialize model
    model = CNN_Net3D(output_features)
    model.to(device)

    # 8. Define loss and optimizer
    loss_f = LossCEL(device = device)
    optimizer = optim.Adam(model.parameters(), lr = config['lr'],
                            weight_decay = config['weight_decay'])

    # 9. Train model
    print('Training model in batches of {}..'.format(config['batch_size']))

    agent = NetAgent(model = model, device = device)
    losses_train, losses_cum_train, losses_val, losses_cum_val,\
    accuracy_train, accuracy_det_train, accuracy_val,\
    accuracy_det_val = agent.train(optimizer, loss_f, dl,
                 dl_val, nr_epochs = config['nr_epochs'],
                                       save_path = paths,
                 save_interval = config['save_interval'],
                             msg_bot = config['msg_bot'],
           bot_msg_interval = config['bot_msg_interval'])
                        
    # 10. Build test dataloader
    dl = DataLoader(datasets[(test_ds)], 
            batch_size = config['batch_size'], shuffle = True)
    
    # 11. Test model
    print('Testing model in batches of {}..'.format(config['batch_size']))
    losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot = config['msg_bot'])

    # 12. Save results
    save_results(model, config['noise'], paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val)
    

def _CNN_restore_and_train(config):
    r"""This function loads an existing state based on the config file, trains
        the missing epochs and saves the result."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['num_intensities']
    dataset_name = config['train_on'][0]

    # 2. Define data to restore dataset
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], gpu=True, cuda=config['device'], msg_bot = config['msg_bot'],\
                     nr_images=config['nr_images'], build_dataset=True, dtype='train', noise=config['noise'],\
                     ds_name=dataset_name)

    data.add_dataset(JIP)
    train_ds = (dataset_name, 'train')
    val_ds = (dataset_name, 'val')
    test_ds = (dataset_name, 'test')

    # 3. Restore and define path
    paths = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'states')
    pathr = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'results')
    splits = lr.load_json(path=paths, name='data_splits')
    print('Restored existing splits')

    # 4. Create data splits for each repetition
    print('Bring data to PyTorch format..')
    
    # 5. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in splits[ds_name][0].items():
            if len(data_ixs) > 0: # Sometimes val indicess may be an empty list
                aug = config['augment_strat'] if not('test' in split) else 'none'

                # --- Remove when using right model --> Only for 2D dummy! --- #
                datasets[(ds_name, split)] = PytorchCNN2DDataset(ds, 
                    ix_lst = data_ixs, size = (1, 299, 299), aug_key = aug, 
                    resize = False)
                # --- Remove when using right model --> Only for 2D dummy! --- #
                
                """
                datasets[(ds_name, split)] = Pytorch3DQueue(ds, 
                    ix_lst = data_ixs, size = (1, 299, 299, 10), aug_key = aug, 
                    samples_per_volume = 10)"""

    # 6. Build train dataloader
    dl = DataLoader(datasets[(train_ds)], 
        batch_size = config['batch_size'], shuffle = True,
        num_workers = 1)
    dl_val = DataLoader(datasets[(val_ds)], 
        batch_size = config['batch_size'], shuffle = True,
        num_workers = 1)

    # 7. Initialize model
    model = CNN_Net2D(output_features) 
    model.to(device)

    # 8. Define loss and optimizer
    loss_f = LossCEL(device = device)
    optimizer = optim.Adam(model.parameters(), lr = config['lr'],
                            weight_decay = config['weight_decay'])

    # 9. Train model
    state_names = [name for name in os.listdir(paths) if '.' not in name]
    state_name = state_names[0].split('_')[0]
    for idx, state in enumerate(state_names):
        state_names[idx] = int(state.split('_')[-1])
    state_names.sort()
    state_name += '_' + str(state_names[-1])

    print('Restore last saved model from epoch {}..'.format(state_name.split('_')[-1]))
    agent = NetAgent(model = model, device = device)
    restored, restored_results = agent.restore_state(paths, state_name, optimizer = optimizer)
    if not restored:
        print("Desired state could not be recovered. --> Error!")
        raise FileNotFoundError

    losses_train_r, losses_cum_train_r, losses_val_r, losses_cum_val_r, accuracy_train_r,\
    accuracy_det_train_r, accuracy_val_r, accuracy_det_val_r = restored_results

    print('Training model in batches of {}..'.format(config['batch_size']))
    losses_train, losses_cum_train, losses_val, losses_cum_val,\
    accuracy_train, accuracy_det_train, accuracy_val,\
    accuracy_det_val = agent.train(optimizer, loss_f, dl,
                    dl_val, nr_epochs=config['nr_epochs'],
             start_epoch = int(state_name.split('_')[-1]),
      save_path = paths, losses = losses_train_r.tolist(),
                 losses_cum = losses_cum_train_r.tolist(),
                       losses_val = losses_val_r.tolist(),
               losses_cum_val = losses_cum_val_r.tolist(),
                     accuracy = accuracy_train_r.tolist(),
        accuracy_detailed = accuracy_det_train_r.tolist(),
                   accuracy_val = accuracy_val_r.tolist(),
      accuracy_val_detailed = accuracy_det_val_r.tolist(),
                  save_interval = config['save_interval'],
                              msg_bot = config['msg_bot'],
            bot_msg_interval = config['bot_msg_interval'])

    # 10. Build test dataloader
    dl = DataLoader(datasets[(test_ds)], 
            batch_size = config['batch_size'], shuffle = True)
    
    # 11. Test model
    print('Testing model in batches of {}..'.format(config['batch_size']))
    losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot = config['msg_bot'])

    # 12. Save results
    save_results(model, config['noise'], paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val)


def _CNN_retrain(config):
    r"""This function loads a pre-trained model from presistent given the noise
        and retrains this model using tranfer learning."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['num_intensities']
    dataset_name = 'retrain'

    # 2. Define data --> Extra in JIP_dataset that loads everything from preprocessed for train!
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], gpu=True, cuda=config['device'], msg_bot = config['msg_bot'],\
                     nr_images=config['nr_images'], build_dataset=True, dtype='train', noise=config['noise'],\
                     ds_name=dataset_name)

    data.add_dataset(JIP)
    train_ds = (dataset_name, 'train')
    val_ds = (dataset_name, 'val')
    test_ds = (dataset_name, 'test')

    # 3. Split data and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio = config['test_ratio'], 
                          val_ratio = config['val_ratio'], nr_repetitions = 1, cross_validation = False)
    paths = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'states')
    pathr = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'results')
    if not os.path.exists(paths):
        os.makedirs(paths)
    else:
        # Empty directory
        shutil.rmtree(paths)
        os.makedirs(paths)
    if not os.path.exists(pathr):
        os.makedirs(pathr)
    else:
        # Empty directory
        shutil.rmtree(pathr)
        os.makedirs(pathr)

    # Save split
    if splits is not None:
        lr.save_json(splits, path = paths, name = 'data_splits')

    # 4. Create data splits for each repetition
    print('Bring data to PyTorch format..')

    # 5. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in splits[ds_name][0].items():
            if len(data_ixs) > 0: # Sometimes val indices may be an empty list
                aug = config['augment_strat'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = Pytorch3DQueue(ds, 
                    ix_lst = data_ixs, size = (1, 299, 299, 10), aug_key = aug, 
                    samples_per_volume = 10)

    # 6. Build train dataloader
    dl = DataLoader(datasets[(train_ds)], 
        batch_size = config['batch_size'], shuffle = True,
        num_workers = 1)
    dl_val = DataLoader(datasets[(val_ds)], 
        batch_size = config['batch_size'], shuffle = True,
        num_workers = 1)

    # 7. Load pre-trained model
    model = CNN_Net3D(output_features)
    state_dict = torch.load(os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], config['noise'], 'model_state_dict.zip'))
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # 8. Define loss and optimizer
    loss_f = LossCEL(device = device)
    optimizer = optim.Adam(model.parameters(), lr = config['lr'],
                            weight_decay = config['weight_decay'])

    # 9. Train model
    print('Training model in batches of {}..'.format(config['batch_size']))

    agent = NetAgent(model = model, device = device)
    losses_train, losses_cum_train, losses_val, losses_cum_val,\
    accuracy_train, accuracy_det_train, accuracy_val,\
    accuracy_det_val = agent.train(optimizer, loss_f, dl,
                 dl_val, nr_epochs = config['nr_epochs'],
                                       save_path = paths,
                 save_interval = config['save_interval'],
                             msg_bot = config['msg_bot'],
           bot_msg_interval = config['bot_msg_interval'])
                        
    # 10. Build test dataloader
    dl = DataLoader(datasets[(test_ds)], 
            batch_size = config['batch_size'], shuffle = True)
    
    # 11. Test model
    print('Testing model in batches of {}..'.format(config['batch_size']))
    losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot = config['msg_bot'])

    # 12. Save results
    save_results(model, config['noise'], paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val)


# -------------------------
# EWC approach
# -------------------------
def _CNN_initialize_and_train_ewc(config, retrain):
    r"""This function selects random images etc. based on the config file
        and starts training the model. If everything works fine, without
        and error, the results will be saved. The Continual Learning method
        EWC will be used. This can also be used to retrain the model if the
        already trained on datasets are available.
        NOTE: retrain indicates if a previously trained model will be used for
              retraining. This is only the case when institutes train pre-trained
              models with their own data and the data on which the models are
              previously trained is available."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['num_intensities']
    dataset_names = config['train_on']
    dataset_names.sort()
    dataset_names = {idx: ds for idx, ds in enumerate(dataset_names)}

    # 2. Define data
    #datasets = list()
    pathss = list()
    pathrs = list()
    dl_trains = dict()
    dl_vals = dict()
    dl_tests = dict()

    # 3. Prepare for EWC train
    for idx, ds in dataset_names.items():
        print('Build dataset \'{}\'..'.format(ds))
        data = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                          augmentation=config['augmentation'], gpu=True, cuda=config['device'], msg_bot = config['msg_bot'],\
                          nr_images=config['nr_images'], build_dataset=True, dtype='train', noise=config['noise'],\
                          ds_name=ds)
        #datasets.append(data)
        train_ds = (ds, 'train')
        val_ds = (ds, 'val')
        test_ds = (ds, 'test')

        pathss.append(os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'states', ds))
        pathrs.append(os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'results', ds))

        print('Split \'{}\' dataset into train - val - test sets..'.format(ds))
        splits = dict()
        for ds_name, ds in data.datasets.items():
            splits[ds_name] = split_dataset(ds, test_ratio = config['test_ratio'],\
                              val_ratio = config['val_ratio'], nr_repetitions = 1, cross_validation = False)
        paths = pathss[idx]
        pathr = pathrs[idx]
        if not os.path.exists(paths):
            os.makedirs(paths)
        else:
            # Empty directory
            shutil.rmtree(paths)
            os.makedirs(paths)
        if not os.path.exists(pathr):
            os.makedirs(pathr)
        else:
            # Empty directory
            shutil.rmtree(pathr)
            os.makedirs(pathr)

        # Save split
        if splits is not None:
            lr.save_json(splits, path = paths, name = 'data_splits')
            
        print('Bring \'{}\' dataset to PyTorch format..'.format(ds))
        dataset = dict()
        for ds_name, ds in data.dataset.items():
            for split, data_ixs in splits[ds_name][0].items():
                if len(data_ixs) > 0: # Sometimes val indices may be an empty list
                    aug = config['augment_strat'] if not('test' in split) else 'none'
                    dataset[(ds_name, split)] = Pytorch3DQueue(ds, 
                        ix_lst = data_ixs, size = (1, 299, 299, 10), aug_key = aug, 
                        samples_per_volume = 10)

        print('Build Dataloaders for \'{}\' dataset..'.format(ds))
        dl_trains[idx] = DataLoader(dataset[(train_ds)], 
            batch_size = config['batch_size'], shuffle = True,
            num_workers = 1)
        dl_vals[idx] = DataLoader(dataset[(val_ds)], 
            batch_size = config['batch_size'], shuffle = True,
            num_workers = 1)
        dl_tests[idx] = DataLoader(dataset[(test_ds)], 
                batch_size = config['batch_size'], shuffle = True)

    # 4. Train each task after another using EWC to avoid forgetting
    trained_on = dict()
    path = os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], config['noise'])
    for task in range(len(dataset_names)):
        print("Train on dataset \'{}\' --> Task {} of {}..".format(dataset_names[task], task+1, len(dataset_names)+1))
        #data = Data()
        #data.add_dataset(datasets[task])
        #train_ds = (dataset_names[task], 'train')
        #val_ds = (dataset_names[task], 'val')
        #test_ds = (dataset_names[task], 'test')

        # Update trained_on dict
        trained_on['start_to_train_on'] = dataset_names[task]
        lr.save_json_beautiful(trained_on, path, 'model_trained_on_ds')

        # 4.1 Define path
        paths = pathss[task]
        pathr = pathrs[task]

        # 4.2 Load train dataloader
        dl = dl_trains[task]
        dl_val = dl_vals[task]

        # 4.3 Initialize model
        if task == 0 and not retrain:
            model = lr.load_model(CNN_Net3D, output_features, None, False)
        else:   # Model trained on previous datasets needed, so load it
            path = os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], config['noise'], 'model_state_dict.zip')
            model = lr.load_model(CNN_Net3D, output_features, path, True)
        model.to(device)

        # 4.4 Define loss and optimizer
        loss_f = LossCEL(device = device)
        optimizer = optim.Adam(model.parameters(), lr = config['lr'],\
                                weight_decay = config['weight_decay'])

        # 4.5 Train model
        print('Training model in batches of {}..'.format(config['batch_size']))

        if task == 0:
            # Train normally
            print('Training model in batches of {}..'.format(config['batch_size']))
            agent = NetAgent(model = model, device = device)
            losses_train, losses_cum_train, losses_val, losses_cum_val,\
                      accuracy_train, accuracy_det_train, accuracy_val,\
                  accuracy_det_val = agent.train(optimizer, loss_f, dl,\
                               dl_val, nr_epochs = config['nr_epochs'],\
                                                     save_path = paths,\
                               save_interval = config['save_interval'],\
                                           msg_bot = config['msg_bot'],\
                          bot_msg_interval = config['bot_msg_interval'])
        else:
            # Build random old tasks for EWC approach
            old_tasks = list()
            for sub_task in range(task):
                old_tasks = old_tasks + dl_trains[sub_task].dataset.get_sample(int(config['sample_size']/(task)))
            print("Length of old_tasks = {}".format(len(old_tasks)))
            #old_tasks = random.sample(old_tasks, config['sample_size'])
            # Train with EWC method
            ewc = EWC(model, old_tasks, device)
            print('Training model in batches of {}..'.format(config['batch_size']))
            agent = NetAgent(model = model, device = device)
            losses_train, losses_cum_train, losses_val, losses_cum_val,\
                      accuracy_train, accuracy_det_train, accuracy_val,\
            accuracy_det_val = agent.ewc_train(ewc, config['importance'],\
                                           optimizer, loss_f, dl, dl_val,\
                      nr_epochs = config['nr_epochs'], save_path = paths,\
                                 save_interval = config['save_interval'],\
                                             msg_bot = config['msg_bot'],\
                            bot_msg_interval = config['bot_msg_interval'])
                        
        # 4.6 Load test dataloader
        dl = dl_tests[task]
        
        # 4.7 Test model
        print('Testing model in batches of {}..'.format(config['batch_size']))
        losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot = config['msg_bot'])

        # 4.8 Save results and update trained_on dict
        save_results(model, config['noise'], paths, pathr, losses_train, losses_val, accuracy_train,
                     accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                     accuracy_det_test, losses_cum_train, losses_cum_val)
    
        trained_on[task] = dataset_names[task]
        del trained_on['start_to_train_on']
        lr.save_json_beautiful(trained_on, path, 'model_trained_on_ds')


def _CNN_restore_and_train_ewc(config):
    r"""This function loads an existing state based on the config file, trains
        the missing epochs and saves the result. The Continual Learning method
        EWC will be used."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['num_intensities']
    dataset_names = config['train_on']
    dataset_names.sort()
    dataset_names = {idx: ds for idx, ds in enumerate(dataset_names)}

    # 2. Define data
    #datasets = list()
    pathss = list()
    pathrs = list()
    dl_trains = dict()
    dl_vals = dict()
    dl_tests = dict()

    # 3. Prepare for EWC train
    for idx, ds in dataset_names.items():
        print('Build dataset \'{}\'..'.format(ds))
        data = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                          augmentation=config['augmentation'], gpu=True, cuda=config['device'], msg_bot = config['msg_bot'],\
                          nr_images=config['nr_images'], build_dataset=True, dtype='train', noise=config['noise'],\
                          ds_name=ds)
        #datasets.append(data)
        train_ds = (ds, 'train')
        val_ds = (ds, 'val')
        test_ds = (ds, 'test')

        pathss.append(os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'states', ds))
        pathrs.append(os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'results', ds))

        print('Restore split \'{}\' dataset into train - val - test sets..'.format(ds))
        paths = pathss[idx]
        pathr = pathrs[idx]
        splits = lr.load_json(path=paths, name='data_splits')
            
        print('Bring \'{}\' dataset to PyTorch format..'.format(ds))
        dataset = dict()
        for ds_name, ds in data.dataset.items():
            for split, data_ixs in splits[ds_name][0].items():
                if len(data_ixs) > 0: # Sometimes val indices may be an empty list
                    aug = config['augment_strat'] if not('test' in split) else 'none'
                    dataset[(ds_name, split)] = Pytorch3DQueue(ds, 
                        ix_lst = data_ixs, size = (1, 299, 299, 10), aug_key = aug, 
                        samples_per_volume = 10)

        print('Build Dataloaders for \'{}\' dataset..'.format(ds))
        dl_trains[idx] = DataLoader(dataset[(train_ds)], 
            batch_size = config['batch_size'], shuffle = True,
            num_workers = 1)
        dl_vals[idx] = DataLoader(dataset[(val_ds)], 
            batch_size = config['batch_size'], shuffle = True,
            num_workers = 1)
        dl_tests[idx] = DataLoader(dataset[(test_ds)], 
                batch_size = config['batch_size'], shuffle = True)

    # 4. Train each task after another using EWC to avoid forgetting
    path = os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], config['noise'])
    trained_on = lr.load_json(path, 'model_trained_on_ds')

    # Remove tasks on which the model already trained on
    new_ds_names = dataset_names
    for idx, _ in trained_on.items():
        if isinstance(idx, int): # Only tasks are ids, 'start_to_train_on' is not an integer
            del new_ds_names[idx]

    # Train the model
    for task in new_ds_names.keys():
        print("Train on dataset \'{}\' --> Task {} of {}..".format(new_ds_names[task], task+1, len(dataset_names)+1))
        #data = Data()
        #data.add_dataset(datasets[task])
        #train_ds = (dataset_names[task], 'train')
        #val_ds = (dataset_names[task], 'val')
        #test_ds = (dataset_names[task], 'test')

        # Update trained_on dict
        trained_on['start_to_train_on'] = dataset_names[task]
        lr.save_json_beautiful(trained_on, path, 'model_trained_on_ds')

        # 4.1 Define path
        paths = pathss[task]
        pathr = pathrs[task]

        # 4.2 Load train dataloader
        dl = dl_trains[task]
        dl_val = dl_vals[task]

        # 4.3 Initialize model
        if task == 0:
            model = lr.load_model(CNN_Net3D, output_features, None, False)
        else:   # Model trained on previous datasets needed, so load it
            path = os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], config['noise'], 'model_state_dict.zip')
            model = lr.load_model(CNN_Net3D, output_features, path, True)
        model.to(device)

        # 4.4 Define loss and optimizer
        loss_f = LossCEL(device = device)
        optimizer = optim.Adam(model.parameters(), lr = config['lr'],\
                                weight_decay = config['weight_decay'])

        # 4.5 Train model
        print('Training model in batches of {}..'.format(config['batch_size']))

        if task == 0:
            # Train normally
            state_names = [name for name in os.listdir(paths) if '.' not in name]
            state_name = state_names[0].split('_')[0]
            for idx, state in enumerate(state_names):
                state_names[idx] = int(state.split('_')[-1])
            state_names.sort()
            state_name += '_' + str(state_names[-1])

            print('Restore last saved model from epoch {}..'.format(state_name.split('_')[-1]))
            agent = NetAgent(model = model, device = device)
            restored, restored_results = agent.restore_state(paths, state_name, optimizer = optimizer)
            if not restored:
                print("Desired state could not be recovered. --> Error!")
                raise FileNotFoundError
            losses_train_r, losses_cum_train_r, losses_val_r, losses_cum_val_r, accuracy_train_r,\
            accuracy_det_train_r, accuracy_val_r, accuracy_det_val_r = restored_results

            print('Training model in batches of {}..'.format(config['batch_size']))
            agent = NetAgent(model = model, device = device)
            losses_train, losses_cum_train, losses_val, losses_cum_val,\
                      accuracy_train, accuracy_det_train, accuracy_val,\
                  accuracy_det_val = agent.train(optimizer, loss_f, dl,\
                                 dl_val, nr_epochs=config['nr_epochs'],\
                          start_epoch = int(state_name.split('_')[-1]),\
                   save_path = paths, losses = losses_train_r.tolist(),\
                              losses_cum = losses_cum_train_r.tolist(),\
                                    losses_val = losses_val_r.tolist(),\
                            losses_cum_val = losses_cum_val_r.tolist(),\
                                  accuracy = accuracy_train_r.tolist(),\
                     accuracy_detailed = accuracy_det_train_r.tolist(),\
                                accuracy_val = accuracy_val_r.tolist(),\
                   accuracy_val_detailed = accuracy_det_val_r.tolist(),\
                               save_interval = config['save_interval'],\
                                           msg_bot = config['msg_bot'],\
                          bot_msg_interval = config['bot_msg_interval'])
        else:
            # Build random old tasks for EWC approach
            old_tasks = list()
            for sub_task in range(task):
                old_tasks = old_tasks + dl_trains[sub_task].dataset.get_sample(int(config['sample_size']/(task)))
            print("Length of old_tasks = {}".format(len(old_tasks)))
            #old_tasks = random.sample(old_tasks, config['sample_size'])
            # Train with EWC method
            ewc = EWC(model, old_tasks, device)

            state_names = [name for name in os.listdir(paths) if '.' not in name]
            state_name = state_names[0].split('_')[0]
            for idx, state in enumerate(state_names):
                state_names[idx] = int(state.split('_')[-1])
            state_names.sort()
            state_name += '_' + str(state_names[-1])

            print('Restore last saved model from epoch {}..'.format(state_name.split('_')[-1]))
            agent = NetAgent(model = model, device = device)
            restored, restored_results = agent.restore_state(paths, state_name, optimizer = optimizer)
            if not restored:
                print("Desired state could not be recovered. --> Error!")
                raise FileNotFoundError
            losses_train_r, losses_cum_train_r, losses_val_r, losses_cum_val_r, accuracy_train_r,\
            accuracy_det_train_r, accuracy_val_r, accuracy_det_val_r = restored_results

            print('Training model in batches of {}..'.format(config['batch_size']))
            agent = NetAgent(model = model, device = device)
            losses_train, losses_cum_train, losses_val, losses_cum_val,\
                        accuracy_train, accuracy_det_train, accuracy_val,\
            accuracy_det_val = agent.ewc_train(ewc, config['importance'],\
                                           optimizer, loss_f, dl, dl_val,\
                                           nr_epochs=config['nr_epochs'],\
                            start_epoch = int(state_name.split('_')[-1]),\
                     save_path = paths, losses = losses_train_r.tolist(),\
                                losses_cum = losses_cum_train_r.tolist(),\
                                      losses_val = losses_val_r.tolist(),\
                              losses_cum_val = losses_cum_val_r.tolist(),\
                                    accuracy = accuracy_train_r.tolist(),\
                       accuracy_detailed = accuracy_det_train_r.tolist(),\
                                  accuracy_val = accuracy_val_r.tolist(),\
                     accuracy_val_detailed = accuracy_det_val_r.tolist(),\
                                 save_interval = config['save_interval'],\
                                             msg_bot = config['msg_bot'],\
                            bot_msg_interval = config['bot_msg_interval'])
                        
        # 4.6 Load test dataloader
        dl = dl_tests[task]
        
        # 4.7 Test model
        print('Testing model in batches of {}..'.format(config['batch_size']))
        losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot = config['msg_bot'])

        # 4.8 Save results and trained_on dict
        save_results(model, config['noise'], paths, pathr, losses_train, losses_val, accuracy_train,
                     accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                     accuracy_det_test, losses_cum_train, losses_cum_val)
    
        trained_on[task] = new_ds_names[task]
        del trained_on['start_to_train_on']
        lr.save_json_beautiful(trained_on, path, 'model_trained_on_ds')


def _CNN_predict(config):
    r"""This function loads an existing (pre-trained) model and makes predictions based on the input file(s)."""

    # 1. Load data
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], gpu=True, cuda=config['device'], msg_bot = config['msg_bot'],\
                     nr_images=config['nr_images'], build_dataset=True, dtype='inference', noise=config['noise'])
    data.add_dataset(JIP)

    # 2. Load pre-trained models
    NQQ = NoiseQualityQuantifier(device=config['device'], output_features=config['num_intensities'])

    # 3. Calculate metrices
    metrices = dict()
    for num, inst in enumerate(JIP.instances):
        msg = "Loading SimpleITK images and calculating metrices (doing inference): "
        msg += str(num + 1) + " of " + str(len(JIP.instances)) + "."
        print (msg, end = "\r")
        # --- Updating the dictionary results in one lin metrics for all scans in data_dir --> need a key that won't
        # --- be updated. That's why a number/the patient id is used!
        path = os.path.join(os.environ["PREPROCESSED_WORKFLOW_DIR"], os.environ["PREPROCESSED_OPERATOR_OUT_DATA_DIR"], inst.name, 'img', 'img.nii.gz')
        #metrices[num+1] = NQQ.get_quality(x=inst.x.tensor.permute(3, 0, 1, 2), path=path, gpu=True, cuda=config['device'])    # Number to metrics
        metrices[inst.name] = NQQ.get_quality(x=inst.x.tensor.permute(3, 0, 1, 2), path=path, gpu=True, cuda=config['device'])  # Patient Name to metrics

    # 4. Save metrices as json
    out_dir = os.path.join('/', os.environ['WORKFLOW_DIR'], os.environ["OPERATOR_OUT_DIR"])
    lr.save_json_beautiful(metrices, out_dir, 'metrics')


# -------------------------
# Dummy
# -------------------------
def _CNN_initialize_and_train_dummy(config):
    r"""This function selects random images etc. based on the config file
        and starts training the model. If everything works fine, without
        and error, the results will be saved."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['num_intensities']
    dataset_name = config['train_on'][0]

    # 2. Define data
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], gpu=True, cuda=config['device'], msg_bot = config['msg_bot'],\
                     nr_images=config['nr_images'], build_dataset=True, dtype='train', noise=config['noise']\
                     ds_name=dataset_name)

    data.add_dataset(JIP)
    train_ds = (dataset_name, 'train')
    val_ds = (dataset_name, 'val')
    test_ds = (dataset_name, 'test')

    # 3. Split data and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio = config['test_ratio'],
                          val_ratio = config['val_ratio'], nr_repetitions = 1, cross_validation = False)
    paths = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'states')
    pathr = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'results')
    if not os.path.exists(paths):
        os.makedirs(paths)
    else:
        # Empty directory
        shutil.rmtree(paths)
        os.makedirs(paths)
    if not os.path.exists(pathr):
        os.makedirs(pathr)
    else:
        # Empty directory
        shutil.rmtree(pathr)
        os.makedirs(pathr)

    # Save split
    if splits is not None:
        lr.save_json(splits, path = paths, name = 'data_splits')

    # 4. Create data splits for each repetition
    print('Bring data to PyTorch format..')

    # 5. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in splits[ds_name][0].items():
            if len(data_ixs) > 0: # Sometimes val indices may be an empty list
                aug = config['augment_strat'] if not('test' in split) else 'none'

                # --- Remove when using right model --> Only for 2D dummy! --- #
                datasets[(ds_name, split)] = PytorchCNN2DDataset(ds, 
                    ix_lst = data_ixs, size = (1, 299, 299), aug_key = aug, 
                    resize = False)
                # --- Remove when using right model --> Only for 2D dummy! --- #
                
                """
                datasets[(ds_name, split)] = PytorchCNN2DDataset(ds, 
                    ix_lst = data_ixs, size = config['input_shape'], aug_key = aug, 
                    resize = False)"""

    # 6. Build train dataloader
    dl = DataLoader(datasets[(train_ds)], 
        batch_size = config['batch_size'], shuffle = True,
        num_workers = 1)
    dl_val = DataLoader(datasets[(val_ds)], 
        batch_size = config['batch_size'], shuffle = True,
        num_workers = 1)

    # 7. Initialize model
    model = CNN_Net2D(output_features)
    model.to(device)

    # 8. Define loss and optimizer
    loss_f = LossCEL(device = device)
    optimizer = optim.Adam(model.parameters(), lr = config['lr'],
                            weight_decay = config['weight_decay'])

    # 9. Train model
    print('Training model in batches of {}..'.format(config['batch_size']))

    agent = NetAgent(model = model, device = device)
    losses_train, losses_cum_train, losses_val, losses_cum_val,\
    accuracy_train, accuracy_det_train, accuracy_val,\
    accuracy_det_val = agent.train(optimizer, loss_f, dl,\
                 dl_val, nr_epochs = config['nr_epochs'],\
                                       save_path = paths,\
                 save_interval = config['save_interval'],\
                             msg_bot = config['msg_bot'],\
            bot_msg_interval = config['bot_msg_interval'])
                        
    # 10. Build test dataloader
    dl = DataLoader(datasets[(test_ds)], 
            batch_size = config['batch_size'], shuffle = True)
    
    # 11. Test model
    print('Testing model in batches of {}..'.format(config['batch_size']))
    losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot = config['msg_bot'])

    # 12. Save results
    save_results(model, config['noise'], paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val)



#def _CNN_test(config):
    r"""This function loads an existing (pretrained) model and makes predictions based on the input file
        and evaluates the output."""
    """
    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['num_intensities']
    dataset_name = config['train_on'][0]

    # 2. Define data
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], num_intensities=config['num_intensities'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], gpu=True, cuda=config['device'], msg_bot = config['msg_bot'],\
                     nr_images=config['nr_images'], build_dataset=True, dtype='train', noise=config['noise']\
                     ds_name=dataset_name)

    data.add_dataset(JIP)

    # 3. Split data (0% train, 100% test) and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio = 1.0, 
        val_ratio = 0, nr_repetitions = 1, 
        cross_validation = False)
    pathr = os.path.join(os.environ["TRAIN_WORKFLOW_DIR"], os.environ["OPERATOR_OUT_DIR"], config['noise'], 'results')
    if not os.path.exists(pathr):
        os.makedirs(pathr)
    else:
        # Empty directory
        shutil.rmtree(pathr)
        os.makedirs(pathr)

    # 4. Bring data to Pytorch format
    print('Bring data to PyTorch format..')
    
    # 5. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in splits[ds_name][0].items():
            if len(data_ixs) > 0: # Sometimes val indicess may be an empty list
                aug = config['augment_strat'] if not('test' in split) else 'none'

                # --- Remove when using right model --> Only for 2D dummy! --- #
                datasets[(ds_name, split)] = PytorchCNN2DDataset(ds, 
                    ix_lst = data_ixs, size = (1, 299, 299), aug_key = aug, 
                    resize = False)
                # --- Remove when using right model --> Only for 2D dummy! --- #
                
                
                #datasets[(ds_name, split)] = PytorchCNN2DDataset(ds, 
                #    ix_lst = data_ixs, size = config['input_shape'], aug_key = aug, 
                #    resize = False)
        
    # 6. Build test dataloader
    dl = DataLoader(datasets[(test_ds)], 
        batch_size = config['batch_size'], shuffle = True,
        num_workers = 1)

    # 7. Load pretrained model
    model = torch.load(os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], config['noise'], 'model.zip'))
    model.eval()
    model.to(device)

    # 8. Define loss and optimizer
    loss_f = LossCEL(device = device)
    
    # 9. Test model
    agent = NetAgent(model = model, device = device)
    print('Testing model in batches of {}..'.format(config['batch_size']))
    losses_test, _, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot = config['msg_bot'])

    # 10. Save results
    save_only_test_results(config['noise'], pathr, losses_test, accuracy_test, accuracy_det_test)
    """