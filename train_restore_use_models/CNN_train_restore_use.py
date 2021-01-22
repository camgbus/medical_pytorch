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
from mp.data.datasets.ds_mr_lung_decathlon_cnn import DecathlonLung, DecathlonLungRestored
from mp.data.datasets.corona_fra_cnn import FraCoronaDatasetAugmented, FraCoronaDatasetRestored, FraCoronaDataset
from mp.data.datasets.gc_corona_cnn import GCCorona, GCCoronaRestored
from mp.experiments.data_splitting import split_dataset
import mp.utils.load_restore as lr
from mp.data.pytorch.pytorch_cnn_dataset import PytorchCNN2DDataset
from mp.models.cnn.cnn import CNN_Net2D_UKFRA as CNN_Net2D
from mp.eval.losses.losses_cnn import LossCEL
from mp.agents.cnn_agents import NetAgent
from mp.utils.save_results import save_results, save_only_test_results

def CNN_initialize_and_train(config):
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
    output_features = max_likert_value
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
                                       nr_slices=nr_slices))
        train_ds = (dataset_name, 'train')
        val_ds = (dataset_name, 'val')
        test_ds = (dataset_name, 'test')

    if dataset_name == 'UK_FRA':
        dataset_name = 'FRACorona'
        data.add_dataset(FraCoronaDatasetAugmented(augmented=augmented,
                                                  img_size=input_shape,
                                     max_likert_value=max_likert_value,
                                           random_slices=random_slices,
                                      noise=noise, nr_images=nr_images,
                                                   nr_slices=nr_slices,
                                                     set_name='train'))
        train_ds = (dataset_name, 'train')
        val_ds = (dataset_name, 'val')
        test_ds = (dataset_name, 'test')

    if dataset_name == 'GC_Corona':
        data.add_dataset(GCCorona(augmented=augmented,
                                 img_size=input_shape,
                    max_likert_value=max_likert_value,
                          random_slices=random_slices,
                     noise=noise, nr_images=nr_images,
                                 nr_slices=nr_slices))
        train_ds = (dataset_name, 'train')
        val_ds = (dataset_name, 'val')
        test_ds = (dataset_name, 'test')


    # 3. Split data and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio=config['test_ratio'], 
        val_ratio=config['val_ratio'], nr_repetitions=config['nr_runs'], 
        cross_validation=config['cross_validation'])
    paths = os.path.join(storage_data_path, 'models', noise+'_'+dataset_name+'_cnn', 'states')
    pathr = os.path.join(storage_data_path, 'models', noise+'_'+dataset_name+'_cnn', 'results')
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
        # 5. Bring data to Pytorch format
        datasets = dict()
        for ds_name, ds in data.datasets.items():
            for split, data_ixs in splits[ds_name][run_ix].items():
                if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                    aug = config['augmentation'] if not('test' in split) else 'none'
                    datasets[(ds_name, split)] = PytorchCNN2DDataset(ds, 
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
        model = CNN_Net2D(output_features)
        model.to(device)

        # 8. Define loss and optimizer
        loss_f = LossCEL(device=device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                               weight_decay=weight_decay)

        # 9. Train model
        print('Training model in batches of {}..'.format(batch_size))

        agent = NetAgent(model=model, device=device)
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
    save_results(model, noise, 'cnn', dataset_name, paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val)
    

def CNN_restore_and_train(config):
    r"""This function loads an existing state based on the config file, trains
        the missing epochs and saves the result."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    input_shape = config['input_shape']
    batch_size = config['batch_size'] 
    max_likert_value = config['max_likert_value']
    output_features = max_likert_value
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
        train_ds = (dataset_name, 'train')
        val_ds = (dataset_name, 'val')
        test_ds = (dataset_name, 'test')
        
    if dataset_name == 'UK_FRA':
        dataset_name = 'FRACorona'
        data.add_dataset(FraCoronaDatasetRestored(img_size=input_shape,
                                     max_likert_value=max_likert_value,
                                                           noise=noise,
                                                     set_name='train'))
        train_ds = (dataset_name, 'train')
        val_ds = (dataset_name, 'val')
        test_ds = (dataset_name, 'test')

    if dataset_name == 'GC_Corona':
        data.add_dataset(GCCoronaRestored(img_size=input_shape,
                             max_likert_value=max_likert_value,
                                                  noise=noise))
        train_ds = (dataset_name, 'train')
        val_ds = (dataset_name, 'val')
        test_ds = (dataset_name, 'test')


    # 3. Restore and define path
    paths = os.path.join(storage_data_path, 'models', noise+'_'+dataset_name+'_cnn', 'states')
    pathr = os.path.join(storage_data_path, 'models', noise+'_'+dataset_name+'_cnn', 'results')
    splits = lr.load_json(path=paths, name='data_splits')
    print('Restored existing splits')

    # 4. Create data splits for each repetition
    print('Bring data to PyTorch format..')
    # Repeat for each repition
    for run_ix in range(config['nr_runs']):
        # 5. Bring data to Pytorch format
        datasets = dict()
        for ds_name, ds in data.datasets.items():
            for split, data_ixs in splits[ds_name][run_ix].items():
                if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                    aug = config['augmentation'] if not('test' in split) else 'none'
                    datasets[(ds_name, split)] = PytorchCNN2DDataset(ds, 
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
        model = CNN_Net2D(output_features) 
        model.to(device)

        # 8. Define loss and optimizer
        loss_f = LossCEL(device=device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                               weight_decay=weight_decay)

        # 9. Train model
        state_names = [name for name in os.listdir(paths) if '.' not in name]
        state_name = state_names[0].split('_')[0]
        for idx, state in enumerate(state_names):
            state_names[idx] = int(state.split('_')[-1])
        state_names.sort()
        state_name += '_' + str(state_names[-1])

        print('Restore last saved model from epoch {}..'.format(state_name.split('_')[-1]))
        agent = NetAgent(model=model, device=device)
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
    save_results(model, noise, 'cnn', dataset_name, paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val)

def CNN_test(config):
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
                                       nr_slices=nr_slices))
        test_ds = (dataset_name, 'test')

    if dataset_name == 'UK_FRA':
        dataset_name = 'FRACorona'
        """
        # Use for real data (assessment)
        data.add_dataset(FraCoronaDataset(augmented=False,
                                     img_size=input_shape,
                        max_likert_value=max_likert_value,
                            noise=noise, set_name='test'))"""
        # Use for augmented data
        data.add_dataset(FraCoronaDatasetAugmented(augmented=False,
                                              img_size=input_shape,
                                 max_likert_value=max_likert_value,
                                                random_slices=True,
                                  noise=noise, nr_images=nr_images,
                                               nr_slices=nr_slices,
                                                  set_name='test'))
        test_ds = (dataset_name, 'test')
        test_ds = (dataset_name, 'test') 

    if dataset_name == 'GC_Corona':
        data.add_dataset(GCCorona(augmented=False,
                             img_size=input_shape,
                max_likert_value=max_likert_value,
                               random_slices=True,
                 noise=noise, nr_images=nr_images,
                             nr_slices=nr_slices))
        test_ds = (dataset_name, 'test')

    # 3. Split data (0% train, 100% test) and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio=1.0, 
        val_ratio=0, nr_repetitions=config['nr_runs'], 
        cross_validation=config['cross_validation'])
    pathr = os.path.join(storage_data_path, 'models', noise+'_'+dataset_name+'_cnn', 'test_results')
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
        # 5. Bring data to Pytorch format
        datasets = dict()
        for ds_name, ds in data.datasets.items():
            for split, data_ixs in splits[ds_name][run_ix].items():
                if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                    aug = config['augmentation'] if not('test' in split) else 'none'
                    datasets[(ds_name, split)] = PytorchCNN2DDataset(ds, 
                        ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                        resize=config['resize'])
                
        # 6. Build test dataloader, and visualize
        dl = DataLoader(datasets[(test_ds)], 
            batch_size=batch_size, shuffle=True,
            num_workers=1)

        # 7. Load pretrained model
        model = torch.load(os.path.join(storage_data_path, 'models', noise+'_cnn', 'model.zip'))
        model.eval()
        model.to(device)

        # 8. Define loss and optimizer
        loss_f = LossCEL(device=device)
        
        # 9. Test model
        agent = NetAgent(model=model, device=device)
        print('Testing model in batches of {}..'.format(batch_size))
        losses_test, _, accuracy_test, accuracy_det_test = agent.test(loss_f, dl, msg_bot=msg_bot)

    # 10. Save results
    save_only_test_results(noise, pathr, losses_test, accuracy_test, accuracy_det_test)

def CNN_predict(config):
    r"""This function loads an existing (pretrained) model and makes predictions based on the input file."""
    pass