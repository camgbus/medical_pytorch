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
from mp.experiments.data_splitting import split_dataset
import mp.utils.load_restore as lr
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg2DDataset
from mp.eval.losses.losses_cnn import LossCEL
from mp.agents.cnn_agents import NetAgent
from mp.utils.save_results import save_results, save_only_test_results
from mp.data.datasets.corona_fra_seg import UKF2
from mp.models.segmentation.unet_fepegar import UNet2D
from mp.eval.losses.losses_segmentation import LossClassWeighted, LossDiceBCE

def UNet2D_initialize_and_train(config):
    r"""This function selects random images etc. based on the confi file
        and starts training the model. If everything works fine, without
        and error, the results will be saved."""

    # 1. Retrieve information from config dict
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    print('Device name: {}'.format(device_name))
    input_shape = config['input_shape']
    batch_size = config['batch_size'] 
    augmented = config['augmented']
    weight_decay = config['weight_decay']
    save_interval = config['save_interval']
    dataset_name = config['dataset']

    # 2. Define data
    data = Data()
    if dataset_name == 'UKF2':
        data.add_dataset(UKF2(resized=True))
        train_ds = (dataset_name, 'train')
        val_ds = (dataset_name, 'val')
        test_ds = (dataset_name, 'test')
        nr_labels = data.nr_labels
    
    # 3. Split data and define path
    splits = dict()
    for ds_name, ds in data.datasets.items():
        splits[ds_name] = split_dataset(ds, test_ratio=config['test_ratio'], 
        val_ratio=config['val_ratio'], nr_repetitions=config['nr_runs'], 
        cross_validation=config['cross_validation'])
    paths = os.path.join('storage', 'models', 'UNet2D',dataset_name, 'states')
    pathr = os.path.join('storage', 'models', 'UNet2D',dataset_name, 'results')
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
                    datasets[(ds_name, split)] = PytorchSeg2DDataset(ds, 
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
        model = UNet2D(input_shape, nr_labels)
        model.to(device)

        # 8. Define loss and optimizer
        loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=device)
        loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'], 
            device=device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        # 9. Train model
        print('Training model in batches of {}..'.format(batch_size))

        agent = NetAgent(model=model, device=device)
        losses_train, losses_cum_train, losses_val, losses_cum_val = agent.train(optimizer, loss_f, dl,
                       dl_val, nr_epochs=config['nr_epochs'],
                       msg_bot=False, save_path=paths,
                save_interval=save_interval)

        # 10. Build test dataloader, and visualize
        dl = DataLoader(datasets[(test_ds)], 
            batch_size=batch_size, shuffle=True)
        
        # 11. Test model
        print('Testing model in batches of {}..'.format(batch_size))
        losses_test, losses_cum_test = agent.test(loss_f, dl, msg_bot=False)

    # 12. Save results
    save_results(model, 'cnn', dataset_name, paths, pathr, losses_train, losses_val,
                     losses_test, losses_cum_train, losses_cum_val,losses_cum_test)

