# ------------------------------------------------------------------------------
# This file (and it siblings) is explained in more details in
# example_training_scripts.md
# ------------------------------------------------------------------------------

# 1. Imports

import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mp.agents.segmentation_IRMGames_agent import SegmentationIRMGamesAgent
from mp.data.data import Data
from mp.data.datasets.ds_mr_hippocampus_decathlon import DecathlonHippocampus
from mp.data.datasets.ds_mr_hippocampus_dryad import DryadHippocampus
from mp.data.datasets.ds_mr_hippocampus_harp import HarP
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg3DDataset
from mp.eval.losses.losses_segmentation import LossDiceBCE, LossClassWeighted
from mp.eval.result import Result
from mp.experiments.experiment import Experiment
from mp.models.IRMGamesModel import IRMGamesModel
from mp.models.segmentation.unet_fepegar import UNet3D
from mp.utils.early_stopping import EarlyStopping
from mp.utils.load_restore import pkl_dump

warnings.filterwarnings("ignore")

# 2. Define data
data = Data()
decath = DecathlonHippocampus(merge_labels=True)
data.add_dataset(decath)
harp = HarP()
data.add_dataset(harp)
dryad = DryadHippocampus(merge_labels=True)
data.add_dataset(dryad)
nr_labels = data.nr_labels
label_names = data.label_names

# 3. Define configuration
configs = [
    {'experiment_name': 'irm_games_test', 'device': 'cuda:0',
     'nr_runs': 5, 'cross_validation': True, 'val_ratio': 0.1, 'test_ratio': 0.3,
     'input_shape': (1, 48, 64, 64), 'resize': False, 'augmentation': 'hybrid',
     'class_weights': (0., 1.), 'lr': 1e-4, "batch_sizes": [26, 5],
     "nr_epochs": 100,
     "eval_interval": 10,
     "train_ds_names": (decath.name, dryad.name),
     "start": 4
     }
]

# 4. Pre-split datasets to avoid having the "Repetition k i of j" messages spammed at each experiment's start
for config in configs:
    exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)
    exp.set_data_splits(data)

for config in configs:
    print("\n" + config["experiment_name"])

    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    input_shape = config['input_shape']

    # 5. Create experiment directories
    exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)
    train_ds_names = config["train_ds_names"]

    # 6. Create data splits for each repetition
    exp.set_data_splits(data)

    # Now repeat for each repetition
    for run_ix in range(config.get("start", 0), config['nr_runs']):
        exp_run = exp.get_run(run_ix)

        # 7. Bring data to Pytorch format
        datasets = dict()
        for ds_name, ds in data.datasets.items():
            # 2 cases: either the dataset's name is in train_ds_names
            # In which case, we proceed as usual:
            if ds_name in train_ds_names:
                for split, data_ixs in exp.splits[ds_name][exp_run.run_ix].items():
                    if len(data_ixs) > 0:  # Sometimes val indexes may be an empty list
                        aug = config['augmentation'] if not ('test' in split) else 'none'
                        datasets[(ds_name, split)] = PytorchSeg3DDataset(ds,
                                                                         ix_lst=data_ixs, size=input_shape, aug_key=aug,
                                                                         resize=config['resize'])
            # If it's not the case, then the dataset's purpose is only testing and the whole dataset is the test split
            else:
                datasets[(ds_name, "test")] = PytorchSeg3DDataset(ds,
                                                                  ix_lst=None, size=input_shape, aug_key="none",
                                                                  resize=config['resize'])

        # 8. Build train dataloader, and visualize
        dls = [DataLoader(datasets[name, "train"], batch_size=length, shuffle=True)
               for name, length in zip(train_ds_names, config['batch_sizes'])]

        # 9. Initialize model
        ensemble = [UNet3D(input_shape, nr_labels) for _ in train_ds_names]
        model = IRMGamesModel(ensemble,
                              input_shape=input_shape,
                              output_shape=ensemble[0].output_shape,
                              # representation_learner=UNet3D(input_shape, input_shape[0])
                              )
        model.to(device)

        # 10. Define loss and optimizer
        loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=device)
        loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'],
                                   device=device)
        optimizers = [optim.Adam(sub_model.parameters(), lr=config['lr']) for sub_model in model.models]
        early_stopping = EarlyStopping(3, "Mean_ScoreDice[hippocampus]", [name + "_val" for name in train_ds_names])

        # 11. Train model
        results = Result(name='training_trajectory')
        agent = SegmentationIRMGamesAgent(model=model, label_names=label_names, device=device)
        epochs = agent.train_with_early_stopping(results, optimizers, loss_g, train_dataloaders=dls,
                                                 early_stopping=early_stopping,
                                                 init_epoch=0, run_loss_print_interval=config["eval_interval"],
                                                 eval_datasets=datasets, eval_interval=config["eval_interval"],
                                                 save_path=exp_run.paths['states'], save_interval=config["nr_epochs"])

        pkl_dump(epochs, "epochs.pkl", exp_run.paths['obj'])
        # 12. Save and print results for this experiment run
        exp_run.finish(results=results, plot_metrics=['Mean_ScoreDice[hippocampus]'])
