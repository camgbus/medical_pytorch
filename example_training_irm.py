# ------------------------------------------------------------------------------
# The same code as in example_training.ipynp but as a python module instead of 
# jupyter notebook. See that file for more detailed explanations.
# ------------------------------------------------------------------------------

# 1. Imports

import torch
import torch.optim as optim
from mp.agents.segmentation_IRM_agent import SegmentationIRMAgent
from mp.data.data import Data
from mp.data.datasets.ds_mr_hippocampus_decathlon import DecathlonHippocampus
from mp.data.datasets.ds_mr_hippocampus_harp import HarP
from mp.data.datasets.ds_mr_hippocampus_dryad import DryadHippocampus
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg3DDataset
from mp.eval.losses.losses_segmentation import LossDiceBCE, LossClassWeighted
from mp.eval.losses.losses_irm import IRMv1Loss, VRexLoss, MMRexLoss, ERMWrapper
from mp.eval.result import Result
from mp.experiments.experiment import Experiment
from mp.models.segmentation.unet_fepegar import UNet3D
from torch.utils.data import DataLoader
from mp.utils.early_stopping import EarlyStopping

# 2. Define configuration
config = {'experiment_name': 'test_exp', 'device': 'cuda:0',
          'nr_runs': 5, 'cross_validation': True, 'val_ratio': 0.0, 'test_ratio': 0.3,
          'input_shape': (1, 48, 64, 64), 'resize': False, 'augmentation': 'none',
          'class_weights': (0., 1.), 'lr': 2e-4, 'batch_sizes': [27, 5],
          "nr_epochs": 120,
          "penalty_weight": 0, "penalty_anneal_iters": 0,
          "save_interval": 10,
          "loss": "irmv1"
          }

device = config['device']
device_name = torch.cuda.get_device_name(device)
print('Device name: {}'.format(device_name))
input_shape = config['input_shape']

# 3. Create experiment directories
exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)

# 4. Define data
data = Data()
decath = DecathlonHippocampus(merge_labels=True)
data.add_dataset(decath)
harp = HarP()
data.add_dataset(harp)
dryad = DryadHippocampus(merge_labels=True)
data.add_dataset(dryad)
nr_labels = data.nr_labels
label_names = data.label_names

# A tuple of dataset names used for training
train_ds_names = harp.name, dryad.name,  # decath.name

# 5. Create data splits for each repetition
exp.set_data_splits(data)

# Now repeat for each repetition
for run_ix in range(config['nr_runs']):
    exp_run = exp.get_run(run_ix)

    # 6. Bring data to Pytorch format
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

    # 7. Build train dataloader, and visualize
    # ds_lengths = [len(datasets[name, "train"]) for name in train_ds_names]
    # total_length = sum(ds_lengths)
    # dls = [DataLoader(datasets[name, "train"], batch_size=config['batch_size'] * length // total_length, shuffle=True)
    #        for name, length in zip(train_ds_names, ds_lengths)]
    dls = [DataLoader(datasets[name, "train"], batch_size=length, shuffle=True)
           for name, length in zip(train_ds_names, config['batch_sizes'])]

    # 8. Initialize model
    model = UNet3D(input_shape, nr_labels)
    model.to(device)

    # 9. Define loss and optimizer
    erm_loss = LossClassWeighted(LossDiceBCE(bce_weight=1., smooth=1., device=device), weights=config["class_weights"])
    irm_losses = {"vrex": VRexLoss(erm_loss, device=device),
                  "mmrex": MMRexLoss(erm_loss, device=device),
                  "irmv1": IRMv1Loss(erm_loss, device=device),
                  "erm": ERMWrapper(erm_loss, device=device)
                  }
    irm_loss = irm_losses[config["loss"]]

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # 10. Train model
    results = Result(name='training_trajectory')
    agent = SegmentationIRMAgent(model=model, label_names=label_names, device=device)

    # agent.train(results, optimizer, irm_loss, train_dataloaders=dls,
    #             init_epoch=0, nr_epochs=config["nr_epochs"], run_loss_print_interval=config["save_interval"],
    #             eval_datasets=datasets, eval_interval=config["save_interval"],
    #             save_path=exp_run.paths['states'], save_interval=config["nr_epochs"],
    #             penalty_weight=config["penalty_weight"], penalty_anneal_iters=config["penalty_anneal_iters"])

    early_stopping = EarlyStopping(1, "Mean_ScoreDice[hippocampus]", [name + "_test" for name in train_ds_names])
    epochs = agent.train_with_early_stopping(results, optimizer, irm_loss, dls, early_stopping,
                                    init_epoch=0, run_loss_print_interval=config["save_interval"],
                                    eval_datasets=datasets, eval_interval=config["save_interval"],
                                    save_path=exp_run.paths['states'],
                                    penalty_weight=config["penalty_weight"])
    stage1_epoch, stage2_epoch = epochs

    # 11. Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Mean_ScoreDice[hippocampus]'],
                       plot_metrics_args={"axvlines": epochs})

    metric = results.results["Mean_ScoreDice[hippocampus]"][stage2_epoch]
    # Print the mean dice score for each dataset and train/test subset
    for key in metric:
        print(f"{key}: {metric[key]:.3f}")
    # Print the final result in a csv format for easy copy-pasting to the spreadsheet
    print("\t".join(f"{e:.3f}" for e in metric.values()).replace(".", ","))
