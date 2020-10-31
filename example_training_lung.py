# 1. Import needed libraries

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from mp.experiments.experiment import Experiment
from mp.data.data import Data
from mp.data.datasets.ds_mr_lung_decathlon import DecathlonLung
import mp.visualization.visualize_imgs as vis
from mp.data.pytorch.pytorch_reg_dataset import PytorchReg2DDataset
from mp.models.regression.linear_regression import LinearRegression as LinReg
from mp.eval.losses.losses_regression import LossMAE, LossMSE, LossHuber
from mp.eval.result import Result
from mp.utils.load_restore import nifty_dump

# 2. Define configuration dict

config = {'experiment_name':'test_exp_lung', 'device':'cuda:4',
    'nr_runs': 1, 'cross_validation': False, 'val_ratio': 0.0, 'test_ratio': 0.3,
    'input_shape': 3, 'resize': False, 'augmentation': 'none', 
    'lr': 0.0001, 'batch_size': 1, 'output_shape': 1
    }
device = config['device']
device_name = torch.cuda.get_device_name(device)
print('Device name: {}'.format(device_name))
input_shape = config['input_shape']  
output_shape = config['output_shape'] 
batch_size = config['batch_size'] 

# 3. Create experiment directories
#exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)

# 4. Define data
data = Data()
data.add_dataset(DecathlonLung(augmented=True))
train_ds = ('DecathlonLung', 'train')
test_ds = ('DecathlonLung', 'test')

# Get input shape and flatten 3d image using view as described in
# https://towardsdatascience.com/pytorch-layer-dimensions-what-sizes-should-they-be-and-why-4265a41e01fd

model = LinReg(input_shape, output_shape)
#print("The parameters: ", model.state_dict())

"""
# 5. Create data splits for each repetition
exp.set_data_splits(data)

# Repeat for each repetition
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
                    resize=config['resize'])


    # 7. Build train dataloader, and visualize
    dl = DataLoader(datasets[(train_ds)], 
        batch_size=config['batch_size'], shuffle=True)

    # 8. Initialize model
    model = LinReg(input_shape, output_shape)
    model.to(device)
    
    # 9. Define loss and optimizer
    loss = LossMSE(device=device)
    optimizer = optim.SGD(model.parameters(), lr=config['lr'])

    # 10. Train model
    results = Result(name='training_trajectory')   
    agent = RegressionAgent(model=model, device=device)
    agent.train(results, optimizer, loss, train_dataloader=dl,
        init_epoch=0, nr_epochs=10, run_loss_print_interval=5,
        eval_datasets=datasets, eval_interval=5, 
        save_path=exp_run.paths['states'], save_interval=5)

    # 11. Save and print results for this experiment run
    #exp_run.finish(results=results, plot_metrics=['Mean_ScoreDice', 'Mean_ScoreDice[left atrium]'])
    #test_ds_key = '_'.join(test_ds)
    #metric = 'Mean_ScoreDice[left atrium]'
    #last_dice = results.get_epoch_metric(
    #    results.get_max_epoch(metric, data=test_ds_key), metric, data=test_ds_key)
    #print('Last Dice score for left atrium class: {}'.format(last_dice))
    """