# ------------------------------------------------------------------------------
# This file (and it siblings) is explained in more details in
# example_training_scripts.md
# ------------------------------------------------------------------------------

# 1. Imports

import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mp.agents.segmentation_IRM_agent import SegmentationIRMAgent
from mp.data.data import Data
from mp.data.datasets.ds_mr_hippocampus_decathlon import DecathlonHippocampus
from mp.data.datasets.ds_mr_hippocampus_dryad import DryadHippocampus
from mp.data.datasets.ds_mr_hippocampus_harp import HarP
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg3DDataset
from mp.eval.losses.losses_irm import IRMv1Loss, VRexLoss, MMRexLoss, ERMWrapper
from mp.eval.losses.losses_segmentation import LossDiceBCE, LossClassWeighted
from mp.eval.result import Result
from mp.experiments.experiment import Experiment
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
    {"experiment_name": "decath_dryad_hybrid_aug_no_dompred_irmv1", "device": "cuda:0", "nr_runs": 5,
     "cross_validation": True, "val_ratio": 0.1, "test_ratio": 0.3, "input_shape": [1, 48, 64, 64], "resize": False,
     "augmentation": "hybrid", "class_weights": [0.0, 1.0], "lr": 0.0002, "batch_sizes": [26, 5],
     "penalty_weight": 1, "save_interval": 10, "loss": "irmv1",
     "train_ds_names": ["DecathlonHippocampus", "DryadHippocampus[Modality:T1w][Resolution:Standard]"]}
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
    train_ds_names = config["train_ds_names"]

    # 5. Create experiment directories
    exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)

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
        model = UNet3D(input_shape, nr_labels)
        model.to(device)

        # 10. Define loss and optimizer
        # erm_loss = LossDiceBCE(bce_weight=1., smooth=1., device=device)
        # irm_loss = LossClassWeightedIRM(VRexLoss(erm_loss, device=device), weights=config["class_weights"])
        erm_loss = LossClassWeighted(LossDiceBCE(bce_weight=1., smooth=1., device=device),
                                     weights=config["class_weights"])

        irm_losses = {"vrex": VRexLoss(erm_loss, device=device),
                      "mmrex": MMRexLoss(erm_loss, device=device),
                      "irmv1": IRMv1Loss(erm_loss, device=device),
                      "erm": ERMWrapper(erm_loss, device)
                      }

        irm_loss = irm_losses[config["loss"]]

        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        # 11. Train model
        results = Result(name='training_trajectory')

        agent = SegmentationIRMAgent(model=model, label_names=label_names, device=device, verbose=True)

        early_stopping = EarlyStopping(1, "Mean_ScoreDice[hippocampus]", [name + "_val" for name in train_ds_names])
        epochs = agent.train_with_early_stopping(results, optimizer, irm_loss, dls, early_stopping,
                                                 init_epoch=0, run_loss_print_interval=config["save_interval"],
                                                 eval_datasets=datasets, eval_interval=config["save_interval"],
                                                 save_path=exp_run.paths['states'],
                                                 penalty_weight=config["penalty_weight"])
        pkl_dump(epochs, "epochs.pkl", exp_run.paths['obj'])

        # 12. Save and print results for this experiment run
        exp_run.finish(results=results, plot_metrics=['Mean_ScoreDice[hippocampus]'],
                       plot_metrics_args={"axvlines": epochs})
