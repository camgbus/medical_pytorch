# 1. Import needed libraries

import torch
import os
import numpy as np
import pandas as pd
from mp.paths import storage_data_path, telegram_login
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.data.data import Data
from mp.data.datasets.ds_mr_lung_decathlon_cnn import DecathlonLung
from mp.experiments.data_splitting import split_dataset
import mp.utils.load_restore as lr
from mp.data.pytorch.pytorch_cnn_dataset import PytorchCNN2DDataset
from mp.models.cnn.cnn import CNN_Net2D
from mp.eval.losses.losses_cnn import LossCEL
from mp.agents.cnn_agents import NetAgent
from mp.visualization.plot_results import plot_numpy


# 2. Define configuration dict
config = {'device':'cuda:4', 'nr_runs': 1, 'cross_validation': False, 
          'val_ratio': 0.2, 'test_ratio': 0.2, 'input_shape': (1, 299, 299),
          'resize': False, 'augmentation': 'none', 'lr': 0.001, 'batch_size': 64,
          'max_likert_value': 5, 'nr_epochs': 300
         }
device = config['device']
device_name = torch.cuda.get_device_name(device)
print('Device name: {}'.format(device_name))
input_shape = config['input_shape']
batch_size = config['batch_size'] 
max_likert_value = config['max_likert_value']
output_features = max_likert_value


# 3. Define data
data = Data()
data.add_dataset(DecathlonLung(augmented=True, img_size=input_shape,
                 max_likert_value=max_likert_value, random_slices=True,
                 noise='spike', nr_images=150, nr_slices=20,
                 original_perc_data=1/max_likert_value))
train_ds = ('DecathlonLung', 'train')
val_ds = ('DecathlonLung', 'val')
test_ds = ('DecathlonLung', 'test')


# 4. Split data and define path
splits = dict()
for ds_name, ds in data.datasets.items():
    splits[ds_name] = split_dataset(ds, test_ratio=config['test_ratio'], 
    val_ratio=config['val_ratio'], nr_repetitions=config['nr_runs'], 
    cross_validation=config['cross_validation'])
paths = os.path.join(storage_data_path, 'models', 'spike', 'states')
pathr = os.path.join(storage_data_path, 'models', 'spike', 'results')
if not os.path.exists(paths):
    os.makedirs(paths)
if not os.path.exists(pathr):
    os.makedirs(pathr)


# 5. Create data splits for each repetition
print('Bring data to PyTorch format..')
# Repeat for each repition
for run_ix in range(config['nr_runs']):
    # 6. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in splits[ds_name][run_ix].items():
            if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                aug = config['augmentation'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = PytorchCNN2DDataset(ds, 
                    ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                    resize=config['resize'])

    # 7. Build train dataloader, and visualize
    dl = DataLoader(datasets[(train_ds)], 
        batch_size=batch_size, shuffle=True,
        num_workers=1)
    dl_val = DataLoader(datasets[(val_ds)], 
        batch_size=batch_size, shuffle=True,
        num_workers=1)

    # 8. Initialize model
    model = CNN_Net2D(output_features)
    model.to(device)

    # 9. Define loss and optimizer
    loss_f = LossCEL(device=device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0.75)

    # 10. Train model
    print('Training model in batches of {}..'.format(batch_size))

    agent = NetAgent(model=model, device=device)
    losses_train, losses_cum_train, losses_cum_val, accuracy_train,\
    accuracy_det_train, accuracy_val, accuracy_det_val = agent.\
                                    train(optimizer, loss_f, dl,
                          dl_val, nr_epochs=config['nr_epochs'],
                              save_path=paths, save_interval=25,
                                            bot_msg_interval=20)

    # 11. Build test dataloader, and visualize
    dl = DataLoader(datasets[(test_ds)], 
        batch_size=batch_size, shuffle=True)
    
    # 12. Test model
    print('Testing model in batches of {}..'.format(batch_size))
    losses_cum_test, accuracy_test, accuracy_det_test = agent.test(loss_f, dl)
    

# 13. Save results
print('Save trained model and losses..')
torch.save(model.state_dict(), os.path.join(paths, 'model_state_dict.zip'))
torch.save(model, os.path.join(storage_data_path, 'models', 'spike', 'model.zip'))
np.save(os.path.join(pathr, 'losses_train.npy'), np.array(losses_train))
np.save(os.path.join(pathr, 'losses_validation.npy'), np.array(losses_cum_val))
np.save(os.path.join(pathr, 'accuracy_train.npy'), np.array(accuracy_train))
np.save(os.path.join(pathr, 'accuracy_detailed_train.npy'), np.array(accuracy_det_train))
np.save(os.path.join(pathr, 'accuracy_validation.npy'), np.array(accuracy_val))
np.save(os.path.join(pathr, 'accuracy_detailed_validation.npy'), np.array(accuracy_det_val))
np.save(os.path.join(pathr, 'losses_test.npy'), np.array(losses_cum_test))
np.save(os.path.join(pathr, 'accuracy_test.npy'), np.array(accuracy_test))
np.save(os.path.join(pathr, 'accuracy_detailed_test.npy'), np.array(accuracy_det_test))
plot_numpy(pd.DataFrame(losses_cum_train, columns =['Epoch', 'Loss']),
    save_path=pathr, save_name='losses_train', title='Losses [train dataset]',
    x_name='Epoch', y_name='Loss', ending='.png', ylog=False, figsize=(10,5),
    xints=float, yints=float)
plot_numpy(pd.DataFrame(accuracy_train, columns =['Epoch', 'Accuracy']),
    save_path=pathr, save_name='accuracy_train', title='Accuracy [train dataset] in %',
    x_name='Epoch', y_name='Accuracy', ending='.png', ylog=False, figsize=(10,5),
    xints=float, yints=float)
plot_numpy(pd.DataFrame(losses_cum_val, columns =['Epoch', 'Loss']),
    save_path=pathr, save_name='losses_val', title='Losses [validation dataset]',
    x_name='Epoch', y_name='Loss', ending='.png', ylog=False, figsize=(10,5),
    xints=float, yints=float)
plot_numpy(pd.DataFrame(accuracy_val, columns =['Epoch', 'Accuracy']),
    save_path=pathr, save_name='accuracy_val', title='Accuracy [validation dataset] in %',
    x_name='Epoch', y_name='Accuracy', ending='.png', ylog=False, figsize=(10,5),
    xints=float, yints=float)
plot_numpy(pd.DataFrame(losses_cum_test, columns =['Datapoints', 'Loss']),
    save_path=pathr, save_name='losses_test', title='Losses [test dataset]',
    x_name='Datapoints', y_name='Loss', ending='.png', ylog=False, figsize=(10,5),
    xints=float, yints=float)
plot_numpy(pd.DataFrame(accuracy_test, columns =['Datapoints', 'Accuracy']),
    save_path=pathr, save_name='accuracy_test', title='Accuracy [test dataset] in %',
    x_name='Datapoints', y_name='Accuracy', ending='.png', ylog=False, figsize=(10,5),
    xints=int, yints=int)