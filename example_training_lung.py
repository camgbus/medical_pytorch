# 1. Import needed libraries

import torch
import os
import numpy as np
from mp.paths import storage_data_path
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.experiments.experiment import Experiment
from mp.data.data import Data
from mp.data.datasets.ds_mr_lung_decathlon import DecathlonLung
from mp.experiments.data_splitting import split_dataset
import mp.utils.load_restore as lr
from mp.data.pytorch.pytorch_reg_dataset import PytorchReg2DDataset
from mp.models.regression.linear_regression import LinearRegression as LinReg
from mp.eval.losses.losses_regression import LossMAE, LossMSE, LossHuber
from mp.agents.regression_agent import RegressionAgent
from mp.eval.result import Result
from mp.utils.load_restore import nifty_dump
from mp.visualization.plot_results import plot_results

# 2. Define configuration dict

config = {'experiment_name':'exp_lung', 'device':'cuda:4',
    'nr_runs': 1, 'cross_validation': False, 'val_ratio': 0.0, 'test_ratio': 0.3,
    'input_shape': (1, 256, 256), 'resize': False, 'augmentation': 'none', 
    'lr': 0.0001, 'batch_size': 1, 'max_likert_value': 5, 'nr_epochs': 2
    }
device = config['device']
device_name = torch.cuda.get_device_name(device)
print('Device name: {}'.format(device_name))
input_shape = config['input_shape']
batch_size = config['batch_size'] 
input_features = int(input_shape[1]*input_shape[2]/batch_size)
output_features = batch_size
max_likert_value = config['max_likert_value']

# 3. Create experiment directories
#exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)

# 4. Define data
data = Data()
data.add_dataset(DecathlonLung(augmented=True, img_size=(256,256), max_likert_value=max_likert_value))
train_ds = ('DecathlonLung', 'train')
test_ds = ('DecathlonLung', 'test')


# 5. Split data and define path
splits = dict()
for ds_name, ds in data.datasets.items():
    splits[ds_name] = split_dataset(ds, test_ratio=config['test_ratio'], 
    val_ratio=config['val_ratio'], nr_repetitions=config['nr_runs'], 
    cross_validation=config['cross_validation'])
paths = os.path.join(os.path.join(storage_data_path, 'model'), 'states')
pathr = os.path.join(os.path.join(storage_data_path, 'model'), 'results')
if not os.path.exists(paths):
    os.makedirs(paths)
if not os.path.exists(pathr):
    os.makedirs(pathr)

# 6. Create data splits for each repetition
#exp.set_data_splits(data)

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


    # 7. Build train dataloader, and visualize
    dl = DataLoader(datasets[(train_ds)], 
        batch_size=config['batch_size'], shuffle=True)

    # 8. Initialize model
    model = LinReg(input_features, output_features,
                   batch_size=config['batch_size'])
    model.to(device)

    # 9. Define loss and optimizer
    loss_f = LossMSE(device=device)
    optimizer = optim.SGD(model.parameters(), lr=config['lr'])

    # 10. Train model
    print('Train model..')
    losses = list()
    for epoch in range(config['nr_epochs']):
        msg = "Running epoch "
        msg += str(epoch + 1) + " of " + str(config['nr_epochs']) + "."
        print (msg, end = "\r")
        epoch_loss = list()
        for idx, (x, y) in enumerate(dl):
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            optimizer.zero_grad()
            loss = loss_f(yhat, y)
            epoch_loss.append(loss)
            loss.backward()
            optimizer.step()
        losses.append(epoch_loss)
        print('Epoch --> Loss: {} --> {}.'.format(epoch, sum(epoch_loss)/len(epoch_loss)))

    """
    results = Result(name='train_noise_regression_model')   
    agent = RegressionAgent(model=model, device=device)
    agent.train(results, optimizer, loss, train_dataloader=dl,
        init_epoch=0, nr_epochs=1, run_loss_print_interval=1,
        eval_datasets=datasets, eval_interval=1, 
        save_path=paths, save_interval=1)"""
    #agent.train(results, optimizer, loss, train_dataloader=dl,
    #    init_epoch=0, nr_epochs=10, run_loss_print_interval=5,
    #    eval_datasets=datasets, eval_interval=5, 
    #    save_path=exp_run.paths['states'], save_interval=5)

    #print('Save results..')
    # 11. Save and print results for the current run
    #lr.pkl_dump(results, path=pathr, name='results')
    #if isinstance(results, list):
    #    for result in results:
    #        plot_results(result, save_path=pathr, measures='Mean_SquaredError')
    #    else:
    #        plot_results(results, save_path=pathr, measures='Mean_SquaredError')
    

print('Save model and losses..')
torch.save(model.state_dict(), os.path.join(storage_data_path, 'model', 'model_state_dict.zip'))
torch.save(model, os.path.join(storage_data_path, 'model', 'model.zip'))
np.save(os.path.join(pathr, 'losses.npy'), np.array(losses))     

"""    # 11. Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Mean_Squared_Error'])
    test_ds_key = '_'.join(test_ds)
    metric = 'Mean_Squared_Error'
    last_MSE = results.get_epoch_metric(
        results.get_max_epoch(metric, data=test_ds_key), metric, data=test_ds_key)
    print('Last MSE for lung class: {}'.format(last_MSE))


# Get input shape and flatten 3d image using view as described in
# https://towardsdatascience.com/pytorch-layer-dimensions-what-sizes-should-they-be-and-why-4265a41e01fd


#print("The parameters: ", model.state_dict())

# Now repeat for each repetition
for run_ix in range(config['nr_runs']):
    exp_run = exp.get_run(run_ix=0)

    # 6. Bring data to Pytorch format
    datasets = dict()
    for ds_name, ds in data.datasets.items():
        for split, data_ixs in exp.splits[ds_name][exp_run.run_ix].items():
            if len(data_ixs) > 0: # Sometimes val indexes may be an empty list
                aug = config['augmentation'] if not('test' in split) else 'none'
                datasets[(ds_name, split)] = PytorchReg2DDataset(ds, 
                    ix_lst=data_ixs, size=input_shape, aug_key=aug, 
                    resize=config['resize'])"""