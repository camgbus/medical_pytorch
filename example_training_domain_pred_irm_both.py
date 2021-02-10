# ------------------------------------------------------------------------------
# This file (and it siblings) is explained in more details in
# example_training_scripts.md
# ------------------------------------------------------------------------------

# 1. Imports

import warnings
from itertools import chain

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mp.agents.segmentation_domain_pred_IRM_both_agent import SegmentationDomainPredictionIRMAgent
from mp.data.data import Data
from mp.data.datasets.ds_mr_hippocampus_decathlon import DecathlonHippocampus
from mp.data.datasets.ds_mr_hippocampus_dryad import DryadHippocampus
from mp.data.datasets.ds_mr_hippocampus_harp import HarP
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg3DDataset
from mp.eval.losses.losses_domain_prediction import ConfusionLoss
from mp.eval.losses.losses_irm import IRMv1Loss, VRexLoss, MMRexLoss, ERMWrapper
from mp.eval.losses.losses_segmentation import LossDiceBCE, LossClassWeighted, LossBCE
from mp.eval.result import Result
from mp.experiments.experiment import Experiment
from mp.models.domain_prediction.domain_predictor_segmentation import DomainPredictor3D
from mp.models.domain_prediction.unet_with_domain_pred import UNetWithDomainPred
from mp.models.segmentation.unet_fepegar import UNet3D
from mp.utils.early_stopping import EarlyStopping
from mp.utils.load_restore import pkl_dump
from mp.visualization.plot_results import plot_results

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
    {'experiment_name': 'dompred2_harp_dryad_hybrid_aug_vrex2_0,5', 'device': 'cuda:0',
     'nr_runs': 5, 'cross_validation': True, 'val_ratio': 0.1, 'test_ratio': 0.3,
     'input_shape': (1, 48, 64, 64), 'resize': False, 'augmentation': 'hybrid',
     'class_weights': (0., 1.), 'lr': 2e-4, 'batch_sizes': [13, 3],
     "beta": 10, "penalty_weight": 1e5,
     "loss": "vrex", "eval_interval": 10,
     "train_ds_names": (harp.name, dryad.name),
 },
]


# 4. Pre-split datasets to avoid having the "Repetition k i of j" messages spammed at each experiment's start
for config in configs:
    exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)
    exp.set_data_splits(data)

for config in configs:
    print(config["experiment_name"])
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    input_shape = config['input_shape']
    train_ds_names = config["train_ds_names"]

    # 5. Create experiment directories
    exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)

    # 6. Create data splits for each repetition
    exp.set_data_splits(data)

    # Now repeat for each repetition
    for run_ix in range(config.get("start", 0), config.get("end", config['nr_runs'])):
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
        unet = UNet3D(input_shape, nr_labels)
        unet.to(device)

        domain_predictor = DomainPredictor3D(input_shape, len(train_ds_names), out_channels_first_layer=16)
        domain_predictor.to(device)

        model = UNetWithDomainPred(unet, domain_predictor, input_shape, (2,))
        model.to(device)

        # 10. Define loss and optimizer
        erm_loss = LossClassWeighted(LossDiceBCE(bce_weight=1., smooth=1., device=device),
                                     weights=config["class_weights"], device=device)
        irm_losses = {"vrex": VRexLoss(erm_loss, device=device),
                      "mmrex": MMRexLoss(erm_loss, device=device),
                      "irmv1": IRMv1Loss(erm_loss, device=device),
                      "erm": ERMWrapper(erm_loss, device=device)
                      }
        loss_f_classifier = irm_losses[config["loss"]]

        erm_loss = LossBCE(device=device)
        irm_losses = {"vrex": VRexLoss(erm_loss, device=device),
                      "mmrex": MMRexLoss(erm_loss, device=device),
                      "irmv1": IRMv1Loss(erm_loss, device=device),
                      "erm": ERMWrapper(erm_loss, device=device)
                      }
        loss_f_domain_predictor = irm_losses[config["loss"]]

        loss_f_encoder = ConfusionLoss()
        losses = [loss_f_classifier, loss_f_domain_predictor, loss_f_encoder]

        optimizer_stage1 = optim.Adam(model.parameters(), lr=config['lr'])
        optimizer_model = optim.Adam(chain(model.encoder_parameters(), model.classifier_parameters()), lr=2e-5)
        optimizer_domain_predictor = optim.Adam(model.domain_predictor_parameters(), lr=2e-6)
        optimizer_encoder = optim.Adam(model.encoder_parameters(), lr=2e-5)
        optimizers = [optimizer_stage1, optimizer_model, optimizer_domain_predictor, optimizer_encoder]

        # 11. Train model
        results = Result(name='training_trajectory')
        agent = SegmentationDomainPredictionIRMAgent(model=model, label_names=label_names, device=device, metrics=["ScoreDice"],
                                                     verbose=True)

        early_stopping = EarlyStopping(1, "Mean_ScoreDice[hippocampus]", [name + "_val" for name in train_ds_names])
        epochs = agent.train_with_early_stopping(results, optimizers, losses, train_dataloaders=dls,
                                                 train_dataset_names=train_ds_names,
                                                 early_stopping=early_stopping,
                                                 run_loss_print_interval=config["eval_interval"],
                                                 eval_datasets=datasets, eval_interval=config["eval_interval"],
                                                 save_path=exp_run.paths['states'],
                                                 alpha=config.get("alpha", 1.0),
                                                 beta=config["beta"], penalty_weight=config["penalty_weight"])
        # Save the stage delimitations in the obj folder
        pkl_dump(epochs, "epochs.pkl", exp_run.paths['obj'])

        # 12. Save and print results for this experiment run
        exp_run.finish(results=results, plot_metrics=['Mean_ScoreDice[hippocampus]'],
                       plot_metrics_args={"axvlines": epochs})
        # Plotting in a separate file, because of seaborn's limited dashes style list
        plot_results(results, save_path=exp_run.paths['results'],
                     save_name="domain_prediction_accuracy.png",
                     measures=["Mean_ScoreAccuracy_DomPred"],
                     axvlines=epochs)