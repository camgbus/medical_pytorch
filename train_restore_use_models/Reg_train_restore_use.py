# Import needed libraries
import torch
import os
import numpy as np
import pandas as pd
import shutil
from mp.paths import storage_data_path
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.data.data import Data
from mp.data.datasets.ds_mr_lung_decathlon_reg import DecathlonLung, DecathlonLungRestored
from mp.data.datasets.corona_fra_reg import FraCoronaDatasetAugmented, FraCoronaDatasetRestored
from mp.experiments.data_splitting import split_dataset
import mp.utils.load_restore as lr
from mp.data.pytorch.pytorch_reg_dataset import PytorchReg2DDataset
from mp.models.regression.linear_regression import LinearRegression as LinReg
from mp.eval.losses.losses_regression import LossMAE, LossMSE, LossHuber
from mp.agents.regression_agent import RegressionAgent
from mp.utils.save_results import save_results, save_only_test_results

def Reg_initialize_and_train(config):
    r"""This function selects random images etc. based on the confi file
        and starts training the model. If everything works fine, without
        and error, the results will be saved."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    input_shape = config['input_shape']
    batch_size = config['batch_size'] 
    max_likert_value = config['max_likert_value']
    output_features = 1
    noise = config['noise']
    augmented = config['augmented']
    random_slices = config['random_slices']
    nr_images = config['nr_images']
    nr_slices = config['nr_slices']
    weight_decay = config['weight_decay']
    save_interval = config['save_interval']
    msg_bot = config['msg_bot']
    bot_msg_interval = config['bot_msg_interval']
    dataset_name = config['dataset']


    # 2. Define data
    data = Data()
    if dataset_name == 'DecathlonLung':
        data.add_dataset(DecathlonLung(augmented=augmented,
                                      img_size=input_shape,
                         max_likert_value=max_likert_value,
                               random_slices=random_slices,
                          noise=noise, nr_images=nr_images,
                                       nr_slices=nr_slices,
                    original_perc_data=1/max_likert_value))
        train_ds = ('DecathlonLung', 'train')
        val_ds = ('DecathlonLung', 'val')
        test_ds = ('DecathlonLung', 'test')

    if dataset_name == 'UK_FRA':
        dataset_name = 'FRACorona'
        data.add_dataset(FraCoronaDatasetAugmented(augmented=augmented,
                                                  img_size=input_shape,
                                     max_likert_value=max_likert_value,
                                           random_slices=random_slices,
                                      noise=noise, nr_images=nr_images,
                                                   nr_slices=nr_slices,
                                                      set_name='train'))
        train_ds = ('FRACorona', 'train')
        val_ds = ('FRACorona', 'val')
        test_ds = ('FRACorona', 'test')


    # 3. Split data and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio=config['test_ratio'], 
        val_ratio=config['val_ratio'], nr_repetitions=config['nr_runs'], 
        cross_validation=config['cross_validation'])
    paths = os.path.join(storage_data_path, 'models', noise+'_'+dataset_name+'_regression', 'states')
    pathr = os.path.join(storage_data_path, 'models', noise+'_'+dataset_name+'_regression', 'results')
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
        lr.save_json(splits, path=paths, name='data_splits')


    # 4. Create data splits for each repetition
    print('Bring data to PyTorch format..')
    # Repeat for each repition
    for run_ix in range(config['nr_runs']):
        # 6. Bring data to Pytorch format
        datasets = dict()
        for ds_name, ds in data.datasets.items():
            for split, data_ixs in splits[ds_name][run_ix].items():
                if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                    aug = config['augmentation'] if not('test' in split) else 'none'
                    datasets[(ds_name, split)] = PytorchReg2DDataset(ds, 
                        ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                        resize=config['resize'])

        # 6. Build train dataloader, and visualize
        dl = DataLoader(datasets[(train_ds)], 
            batch_size=batch_size, shuffle=True,
            num_workers=1)
        dl_val = DataLoader(datasets[(val_ds)], 
            batch_size=batch_size, shuffle=True,
            num_workers=1)

        # 7. Initialize model
        model = LinReg(output_features)
        model.to(device)

        # 8. Define loss and optimizer
        loss_f = LossMAE(device=device)
        optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                              weight_decay=weight_decay)

        # 9. Train model
        print('Training model in batches of {}..'.format(batch_size))

        agent = RegressionAgent(model=model, device=device)
        losses_train, losses_cum_train, losses_val, losses_cum_val,\
        accuracy_train, accuracy_det_train, accuracy_val,\
        accuracy_det_val = agent.train(optimizer, loss_f, dl,
                       dl_val, nr_epochs=config['nr_epochs'],
                                             save_path=paths,
                save_interval=save_interval, msg_bot=msg_bot,
                           bot_msg_interval=bot_msg_interval)

        # 10. Build test dataloader, and visualize
        dl = DataLoader(datasets[(test_ds)], 
            batch_size=batch_size, shuffle=True)
        
        # 11. Test model
        print('Testing model in batches of {}..'.format(batch_size))
        losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot=msg_bot)

    # 12. Save results
    save_results(model, noise, 'regression', dataset_name, paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val)
    

def Reg_restore_and_train(config):
    r"""This function loads an existing state based on the config file, trains
        the missing epochs and saves the result."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    input_shape = config['input_shape']
    batch_size = config['batch_size'] 
    max_likert_value = config['max_likert_value']
    output_features = 1
    noise = config['noise']
    weight_decay = config['weight_decay']
    save_interval = config['save_interval']
    msg_bot = config['msg_bot']
    bot_msg_interval = config['bot_msg_interval']
    dataset_name = config['dataset']


    # 2. Define data to restore dataset
    data = Data()
    if dataset_name == 'DecathlonLung':
        data.add_dataset(DecathlonLungRestored(img_size=input_shape,
                                max_likert_value=max_likert_value,
                                                    noise=noise))
        train_ds = ('DecathlonLung', 'train')
        val_ds = ('DecathlonLung', 'val')
        test_ds = ('DecathlonLung', 'test')

    if dataset_name == 'UK_FRA':
        dataset_name = 'FRACorona'
        data.add_dataset(FraCoronaDatasetRestored(img_size=input_shape,
                                     max_likert_value=max_likert_value,
                                                           noise=noise,
                                                     set_name='train'))
        train_ds = ('FRACorona', 'train')
        val_ds = ('FRACorona', 'val')
        test_ds = ('FRACorona', 'test')


    # 3. Restore and define path
    paths = os.path.join(storage_data_path, 'models', noise+'_'+dataset_name+'_regression', 'states')
    pathr = os.path.join(storage_data_path, 'models', noise+'_'+dataset_name+'_regression', 'results')
    splits = lr.load_json(path=paths, name='data_splits')
    print('Restored existing splits')


    # 4. Create data splits for each repetition
    print('Bring data to PyTorch format..')
    # Repeat for each repition
    for run_ix in range(config['nr_runs']):
        # 6. Bring data to Pytorch format
        datasets = dict()
        for ds_name, ds in data.datasets.items():
            for split, data_ixs in splits[ds_name][run_ix].items():
                if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                    aug = config['augmentation'] if not('test' in split) else 'none'
                    datasets[(ds_name, split)] = PytorchReg2DDataset(ds, 
                        ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                        resize=config['resize'])

        # 6. Build train dataloader, and visualize
        dl = DataLoader(datasets[(train_ds)], 
            batch_size=batch_size, shuffle=True,
            num_workers=1)
        dl_val = DataLoader(datasets[(val_ds)], 
            batch_size=batch_size, shuffle=True,
            num_workers=1)

        # 7. Initialize model
        model = LinReg(output_features)
        model.to(device)

        # 8. Define loss and optimizer
        loss_f = LossMAE(device=device)
        optimizer = optim.SGD(model.parameters(), lr=config['lr'],
                              weight_decay=weight_decay)

        # 9. Train model
        state_names = [name for name in os.listdir(paths) if '.' not in name]
        state_name = state_names[0].split('_')[0]
        for idx, state in enumerate(state_names):
            state_names[idx] = int(state.split('_')[-1])
        state_names.sort()
        state_name += '_' + str(state_names[-1])

        print('Restore last saved model from epoch {}..'.format(state_name.split('_')[-1]))
        agent = RegressionAgent(model=model, device=device)
        restored, restored_results = agent.restore_state(paths, state_name, optimizer=optimizer)
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
                  start_epoch=int(state_name.split('_')[-1]),
             save_path=paths, losses=losses_train_r.tolist(),
                      losses_cum=losses_cum_train_r.tolist(),
                            losses_val=losses_val_r.tolist(),
                    losses_cum_val=losses_cum_val_r.tolist(),
                          accuracy=accuracy_train_r.tolist(),
             accuracy_detailed=accuracy_det_train_r.tolist(),
                        accuracy_val=accuracy_val_r.tolist(),
           accuracy_val_detailed=accuracy_det_val_r.tolist(),
                save_interval=save_interval, msg_bot=msg_bot,
                           bot_msg_interval=bot_msg_interval)

        # 10. Build test dataloader, and visualize
        dl = DataLoader(datasets[(test_ds)], 
            batch_size=batch_size, shuffle=True)
        
        # 11. Test model
        print('Testing model in batches of {}..'.format(batch_size))
        losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot=msg_bot)

    # 12. Save results
    save_results(model, noise, 'regression', dataset_name, paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val)

def Reg_test(config):
    r"""This function loads an existing (pretrained) model and makes predictions based on the input file
        and evaluates the output."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    input_shape = config['input_shape']
    batch_size = config['batch_size'] 
    max_likert_value = config['max_likert_value']
    output_features = max_likert_value
    noise = config['noise']
    augmented = config['augmented']
    random_slices = config['random_slices']
    nr_images = config['nr_images']
    nr_slices = config['nr_slices']
    weight_decay = config['weight_decay']
    msg_bot = config['msg_bot']
    dataset_name = config['dataset']

    # 2. Define data
    data = Data()
    if dataset_name == 'DecathlonLung':
        data.add_dataset(DecathlonLung(augmented=False,
                                  img_size=input_shape,
                     max_likert_value=max_likert_value,
                                    random_slices=True,
                      noise=noise, nr_images=nr_images,
                                   nr_slices=nr_slices,
                original_perc_data=1/max_likert_value))
        test_ds = ('DecathlonLung', 'test')

    if dataset_name == 'UK_FRA':
        dataset_name = 'FRACorona'
        data.add_dataset(FraCoronaDatasetAugmented(augmented=False,
                                              img_size=input_shape,
                                 max_likert_value=max_likert_value,
                                                random_slices=True,
                                  noise=noise, nr_images=nr_images,
                                               nr_slices=nr_slices,
                                                  set_name='test'))
        test_ds = ('FRACorona', 'test')

    # 3. Split data (0% train, 100% test) and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio=1.0, 
        val_ratio=0, nr_repetitions=config['nr_runs'], 
        cross_validation=config['cross_validation'])
    pathr = os.path.join(storage_data_path, 'models', noise+'_'+dataset_name+'_regression', 'test_results')
    if not os.path.exists(pathr):
        os.makedirs(pathr)
    else:
        # Empty directory
        shutil.rmtree(pathr)
        os.makedirs(pathr)

    # 4. Bring data to Pytorch format
    print('Bring data to PyTorch format..')
    # Repeat for each repition
    for run_ix in range(config['nr_runs']):
        # 5. Bring data to Pytorch format, splits are not relevant since all data is used as test
        datasets = dict()
        for ds_name, ds in data.datasets.items():
            for split, data_ixs in splits[ds_name][run_ix].items():
                if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                    aug = config['augmentation'] if not('test' in split) else 'none'
                    datasets[(ds_name, split)] = PytorchReg2DDataset(ds, 
                        ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                        resize=config['resize'])

        # 6. Build test dataloader, and visualize
        dl = DataLoader(datasets[(test_ds)], 
            batch_size=batch_size, shuffle=True,
            num_workers=1)

        # 7. Load pretrained model
        model = torch.load(os.path.join(storage_data_path, 'models', noise+dataset_name+'_regression', 'model.zip'))
        model.eval()
        model.to(device)

        # 8. Define loss and optimizer
        loss_f = LossMAE(device=device)
        
        # 9. Test model
        agent = RegressionAgent(model=model, device=device)
        print('Testing model in batches of {}..'.format(batch_size))
        losses_test, _, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot=msg_bot)

    # 10. Save results
    save_only_test_results(noise, pathr, losses_test, accuracy_test, accuracy_det_test)

def Reg_predict(config):
    r"""This function loads an existing (pretrained) model and makes predictions based on the input file."""
    pass