# Import needed libraries
import torch
import os
import numpy as np
import pandas as pd
from mp.paths import storage_data_path
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.data.data import Data
from mp.data.datasets.ds_mr_lung_decathlon_cnn import DecathlonLung, DecathlonLungRestored
from mp.experiments.data_splitting import split_dataset
import mp.utils.load_restore as lr
from mp.data.pytorch.pytorch_cnn_dataset import PytorchCNN2DDataset
from mp.models.cnn.cnn import CNN_Net2D
from mp.eval.losses.losses_cnn import LossCEL
from mp.agents.cnn_agents import NetAgent
from mp.utils.save_results import save_results

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


    # 2. Define data
    data = Data()
    data.add_dataset(DecathlonLung(augmented=augmented,
                                  img_size=input_shape,
                     max_likert_value=max_likert_value,
                           random_slices=random_slices,
                      noise=noise, nr_images=nr_images,
                                  nr_slices=nr_slices))
    train_ds = ('DecathlonLung', 'train')
    val_ds = ('DecathlonLung', 'val')
    test_ds = ('DecathlonLung', 'test')


    # 3. Split data and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio=config['test_ratio'], 
        val_ratio=config['val_ratio'], nr_repetitions=config['nr_runs'], 
        cross_validation=config['cross_validation'])
    paths = os.path.join(storage_data_path, 'models', noise+'_cnn', 'states')
    pathr = os.path.join(storage_data_path, 'models', noise+'_cnn', 'results')
    if not os.path.exists(paths):
        os.makedirs(paths)
    if not os.path.exists(pathr):
        os.makedirs(pathr)


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
                                save_path=paths, pathr=pathr,
                save_interval=save_interval, msg_bot=msg_bot,
                           bot_msg_interval=bot_msg_interval)

        # 10. Build test dataloader, and visualize
        dl = DataLoader(datasets[(test_ds)], 
            batch_size=batch_size, shuffle=True)
        
        # 11. Test model
        print('Testing model in batches of {}..'.format(batch_size))
        losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl)

    # 12. Save results
    save_results(model, noise, 'cnn', paths, pathr, losses_train, losses_val, accuracy_train,
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


    # 2. Define data to restore dataset
    data = Data()
    data.add_dataset(DecathlonLungRestored(img_size=input_shape,
                              max_likert_value=max_likert_value,
                                                   noise=noise))
    train_ds = ('DecathlonLung', 'train')
    val_ds = ('DecathlonLung', 'val')
    test_ds = ('DecathlonLung', 'test')

    # 3. Split data and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio=config['test_ratio'], 
        val_ratio=config['val_ratio'], nr_repetitions=config['nr_runs'], 
        cross_validation=config['cross_validation'])
    paths = os.path.join(storage_data_path, 'models', noise+'_cnn', 'states')
    pathr = os.path.join(storage_data_path, 'models', noise+'_cnn', 'results')


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
        _, restored_results = agent.restore_state(paths, state_name,
                                         pathr, optimizer=optimizer)
        if restored_results is None:
            print("Desired state could not be recovered. --> Error!")
            raise FileNotFoundError
        losses_train_r, losses_val_r, accuracy_train_r,\
        accuracy_det_train_r, accuracy_val_r, accuracy_det_val_r = restored_results

        print('Training model in batches of {}..'.format(batch_size))
        losses_train, losses_cum_train, losses_val, losses_cum_val,\
        accuracy_train, accuracy_det_train, accuracy_val,\
        accuracy_det_val = agent.train(optimizer, loss_f, dl,
                       dl_val, nr_epochs=config['nr_epochs'],
                  start_epoch=int(state_name.split('_')[-1]),
                                save_path=paths, pathr=pathr,
                save_interval=save_interval, msg_bot=msg_bot,
                           bot_msg_interval=bot_msg_interval)

        # 10. Join data
        losses_cum_train_r = list()
        losses_cum_val_r = list()
        for idx, e_loss in enumerate(losses_train_r):
            losses_cum_train_r.append([idx+1, sum(e_loss) / len(e_loss)])
        for idx, e_loss in enumerate(losses_val_r):
            losses_cum_val_r.append([idx+1, sum(e_loss) / len(e_loss)])

        losses_train_r.tolist().extend(losses_train)
        losses_val_r.tolist().extend(losses_val)
        accuracy_train_r.tolist().extend(accuracy_train)
        accuracy_det_train_r.tolist().extend(accuracy_det_train)
        accuracy_val_r.tolist().extend(accuracy_val)
        accuracy_det_val_r.tolist().extend(accuracy_det_val)
        losses_cum_train_r.extend(losses_cum_train)
        losses_cum_val_r.extend(losses_cum_val)

        # 11. Build test dataloader, and visualize
        dl = DataLoader(datasets[(test_ds)], 
            batch_size=batch_size, shuffle=True)
        
        # 12. Test model
        print('Testing model in batches of {}..'.format(batch_size))
        losses_test, losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl)

    # 13. Save results
    save_results(model, noise, 'cnn', paths, pathr, losses_train_r, losses_val_r, accuracy_train_r,
                 accuracy_det_train_r, accuracy_val_r, accuracy_det_val_r, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train_r, losses_cum_val_r)