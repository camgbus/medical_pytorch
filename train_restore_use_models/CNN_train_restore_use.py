# Import needed libraries
import torch
import os
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.data.data import Data
from mp.data.datasets.dataset_JIP_cnn import JIPDataset
from mp.experiments.data_splitting import split_dataset
import mp.utils.load_restore as lr
from mp.data.pytorch.pytorch_cnn_dataset import PytorchCNN2DDataset
from mp.models.cnn.cnn import CNN_Net2D
from mp.eval.losses.losses_cnn import LossCEL
from mp.agents.cnn_agents import NetAgent
from mp.utils.save_results import save_results, save_only_test_results
import mp.quantifiers.NoiseQualityQuantifier as NoiseQualityQuantifier
from mp.data.DataConnectorJIP import DataConnector

def train_model(config):
    r"""This function tries to intializes and trains a model. It returns True if the model was
        sucessfully trained and if not an error will be returned as well."""
    try:
        _CNN_initialize_and_train(config)
        return True, None
    except: # catch *all* exceptions
        e = traceback.format_exc()
        return False, e

def restore_train_model(config):
    r"""This function tries to restore and trains a model. It returns True if the model was
        sucessfully trained and if not an error will be returned as well."""
    try:
        _CNN_restore_and_train(config)
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

def _CNN_initialize_and_train(config):
    r"""This function selects random images etc. based on the confi file
        and starts training the model. If everything works fine, without
        and error, the results will be saved."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['max_likert_value']
    dataset_name = 'JIP'

    # 2. Define data
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], max_likert_value=config['max_likert_value'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], gpu=True, cuda=config['device'], msg_bot = config['msg_bot'])

    data.add_dataset(JIP.buildDataset('train', config['noise']))
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
            if len(data_ixs) > 0: # Sometimes val indicess may be an empty list
                aug = False if not('test' in split) else 'none'
                datasets[(ds_name, split)] = PytorchCNN2DDataset(ds, 
                    ix_lst = data_ixs, size = input_shape, aug_key = aug, 
                    resize = False)

    # 6. Build train dataloader, and visualize
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
    accuracy_det_val = agent.train(optimizer, loss_f, dl,
                    dl_val, nr_epochs = config['nr_epochs'],
                                        save_path = paths,
                    save_interval = config['save_interval'],
                                msg_bot = config['msg_bot'],
            bot_msg_interval = config['bot_msg_interval'])
                        
    # 10. Build test dataloader, and visualize
    dl = DataLoader(datasets[(test_ds)], 
            batch_size = config['batch_size'], shuffle = True)
    
    # 11. Test model
    print('Testing model in batches of {}..'.format(config['batch_size']))
    losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot = config['msg_bot'])

    # 12. Save results
    save_results(model, noise, paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val)
    

def _CNN_restore_and_train(config):
    r"""This function loads an existing state based on the config file, trains
        the missing epochs and saves the result."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['max_likert_value']
    dataset_name = 'JIP'

    # 2. Define data to restore dataset
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], max_likert_value=config['max_likert_value'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], gpu=True, cuda=config['device'], msg_bot = config['msg_bot'])

    data.add_dataset(JIP.buildDataset('train', config['noise']))
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
                aug = False if not('test' in split) else 'none'
                datasets[(ds_name, split)] = PytorchCNN2DDataset(ds, 
                    ix_lst = data_ixs, size = input_shape, aug_key = aug, 
                    resize = False)

    # 6. Build train dataloader, and visualize
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

    print('Training model in batches of {}..'.format(batch_size))
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
         save_interval = save_interval, msg_bot = msg_bot,
                      bot_msg_interval = bot_msg_interval)

    # 10. Build test dataloader, and visualize
    dl = DataLoader(datasets[(test_ds)], 
            batch_size = config['batch_size'], shuffle = True)
    
    # 11. Test model
    print('Testing model in batches of {}..'.format(config['batch_size']))
    losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot = config['msg_bot'])

    # 12. Save results
    save_results(model, noise, paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val)

def _CNN_predict(config):
    r"""This function loads an existing (pre-trained) model and makes predictions based on the input file(s)."""
    # Build DataConnector and load data
    dc = DataConnector(extension="mhd")
    dc.loadData()

    # Obtain data as ITK and calculate metrices:
    for inst in dc.instances:
        img = inst.getItkImg()
        metrics = NoiseQualityQuantifier.get_quality(x = sitk.GetArrayFromImage(img))
        dc.appendMetric(metrics)

    # Save metrics
    dc.createOutputJson()

#def _CNN_test(config):
    r"""This function loads an existing (pretrained) model and makes predictions based on the input file
        and evaluates the output."""

    """
    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    output_features = config['max_likert_value']
    dataset_name = 'JIP'

    # 2. Define data
    data = Data()
    JIP = JIPDataset(img_size=config['input_shape'], max_likert_value=config['max_likert_value'], data_type=config['data_type'],\
                     augmentation=config['augmentation'], gpu=True, cuda=config['device'], msg_bot = config['msg_bot'])

    data.add_dataset(JIP.buildDataset('train', config['noise']))

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
                aug = False if not('test' in split) else 'none'
                datasets[(ds_name, split)] = PytorchCNN2DDataset(ds, 
                    ix_lst = data_ixs, size = input_shape, aug_key = aug, 
                    resize = False)
        
    # 6. Build test dataloader, and visualize
    dl = DataLoader(datasets[(test_ds)], 
        batch_size = config['batch_size'], shuffle = True,
        num_workers = 1)

    # 7. Load pretrained model
    model = torch.load(os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], noise, 'model.zip'))
    model.eval()
    model.to(device)

    # 8. Define loss and optimizer
    loss_f = LossCEL(device = device)
    
    # 9. Test model
    agent = NetAgent(model = model, device = device)
    print('Testing model in batches of {}..'.format(config['batch_size']))
    losses_test, _, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot = config['msg_bot'])

    # 10. Save results
    save_only_test_results(noise, pathr, losses_test, accuracy_test, accuracy_det_test)
    """