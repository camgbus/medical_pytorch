# ------------------------------------------------------------------------------
# The same code as in example_training.ipynp but as a python module instead of 
# jupyter notebook. See that file for more detailed explanations.
# ------------------------------------------------------------------------------

# 1. Imports

import warnings
from itertools import chain

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mp.agents.segmentation_domain_pred_agent import SegmentationDomainAgent
from mp.data.data import Data
from mp.data.datasets.ds_mr_hippocampus_decathlon import DecathlonHippocampus
from mp.data.datasets.ds_mr_hippocampus_dryad import DryadHippocampus
from mp.data.datasets.ds_mr_hippocampus_harp import HarP
from mp.data.pytorch.pytorch_seg_dataset import PytorchSeg3DDataset
from mp.eval.losses.losses_domain_prediction import ConfusionLoss
from mp.eval.losses.losses_segmentation import LossDiceBCE, LossClassWeighted, LossBCE
from mp.eval.result import Result
from mp.experiments.experiment import Experiment
from mp.models.domain_prediction.domain_predictor_segmentation import DomainPredictor3D
from mp.models.domain_prediction.unet_with_domain_pred import UNetWithDomainPred
from mp.models.segmentation.unet_fepegar import UNet3D
from mp.utils.early_stopping import EarlyStopping

warnings.filterwarnings("ignore")

# 2. Define configuration

# 1: 1e-3, 150, 10, 100, 1e-5, 1e-6, 1e-6
# 2: 1e-4, 150, 10, 100, 1e-5, 1e-6, 1e-6
# 3: 2e-4, 200, 1000, 100, 5e-5, 1e-5, 5e-5
# 4: 2e-4, 160, 10, 120, 5e-5, 1e-5, 1e-4
# 5: 2e-4, 120, 10, 80, 1e-4, 2e-5, 1e-4
# 6: 2e-4, 120, 10, 80, 2e-5, 2e-6, 2e-5
# test_exp_not_decath4_dompred_aug_many: 2e-4, 120, 10, 80, 2e-5, 2e-6, 2e-5
# test_exp_not_decath4_dompred_aug_many2: 2e-4, 120, 10, 80, 2e-5, 2e-6, 2e-6

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

configs = [{'experiment_name': 'a_test', 'device': 'cuda:0',
            'nr_runs': 1, 'cross_validation': False, 'val_ratio': 0.0, 'test_ratio': 0.3,
            'input_shape': (1, 48, 64, 64), 'resize': False, 'augmentation': 'none',
            'class_weights': (0., 1.), 'lr': 2e-4, 'batch_sizes': [13, 3],
            "nr_epochs": 120,
            "beta": 10, "stage1_epochs": 80,
            "save_interval": 10,
            "train_ds_names": (harp.name, dryad.name)
            },

           # {'experiment_name': 'not_harp_crossval5_dompred_mriaug', 'device': 'cuda:0',
           #  'nr_runs': 5, 'cross_validation': True, 'val_ratio': 0.0, 'test_ratio': 0.3,
           #  'input_shape': (1, 48, 64, 64), 'resize': False, 'augmentation': 'mri',
           #  'class_weights': (0., 1.), 'lr': 2e-4, 'batch_sizes': [13, 3],
           #  "nr_epochs": 120,
           #  "beta": 10, "stage1_epochs": 80,
           #  "save_interval": 10,
           #  "train_ds_names": (decath.name, dryad.name)
           #  },
           ]

for config in configs:
    exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)
    exp.set_data_splits(data)

for config in configs:
    print(config["experiment_name"])
    device = config['device']
    device_name = torch.cuda.get_device_name(device)
    # print('Device name: {}'.format(device_name))
    input_shape = config['input_shape']
    train_ds_names = config["train_ds_names"]

    # 3. Create experiment directories
    exp = Experiment(config=config, name=config['experiment_name'], notes='', reload_exp=True)

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
        unet = UNet3D(input_shape, nr_labels)
        unet.to(device)

        domain_predictor = DomainPredictor3D(input_shape, len(train_ds_names), out_channels_first_layer=16)
        domain_predictor.to(device)

        model = UNetWithDomainPred(unet, domain_predictor, input_shape, (2,))
        model.to(device)

        # 9. Define loss and optimizer
        loss_f_classifier = LossClassWeighted(LossDiceBCE(bce_weight=1., smooth=1., device=device),
                                              weights=config["class_weights"])
        loss_f_domain_predictor = LossBCE()
        loss_f_encoder = ConfusionLoss()
        losses = [loss_f_classifier, loss_f_domain_predictor, loss_f_encoder]

        optimizer_stage1 = optim.Adam(model.parameters(), lr=config['lr'])
        optimizer_model = optim.Adam(chain(model.encoder_parameters(), model.classifier_parameters()), lr=2e-5)
        optimizer_domain_predictor = optim.Adam(model.domain_predictor_parameters(), lr=2e-6)
        optimizer_encoder = optim.Adam(model.encoder_parameters(), lr=2e-5)
        optimizers = [optimizer_stage1, optimizer_model, optimizer_domain_predictor, optimizer_encoder]

        # 10. Train model
        results = Result(name='training_trajectory')
        agent = SegmentationDomainAgent(model=model, label_names=label_names, device=device, metrics=["ScoreDice"],
                                        verbose=False)
        # epochs = agent.train(results, optimizers, losses, train_dataloaders=dls,
        #                      init_epoch=0, nr_epochs=config["nr_epochs"],
        #                      run_loss_print_interval=config["save_interval"],
        #                      eval_datasets=datasets, eval_interval=config["save_interval"],
        #                      save_path=exp_run.paths['states'], save_interval=config["nr_epochs"],
        #                      beta=config["beta"], stage1_epochs=config["stage1_epochs"])
        early_stopping = EarlyStopping(1, "Mean_ScoreDice[hippocampus]", [name + "_test" for name in train_ds_names])
        epochs = agent.train_with_early_stopping(results, optimizers, losses, train_dataloaders=dls,
                             early_stopping=early_stopping,
                             run_loss_print_interval=config["save_interval"],
                             eval_datasets=datasets, eval_interval=config["save_interval"],
                             save_path=exp_run.paths['states'], save_interval=config["nr_epochs"],
                             beta=config["beta"])
        stage1_epoch, stage2_epoch, stage3_epoch = epochs

        # 11. Save and print results for this experiment run
        exp_run.finish(results=results, plot_metrics=['Mean_ScoreDice[hippocampus]'])

        # Outputs the results in csv format
        # Stage 1: DICE scores and accuracy scores
        dice = results.results["Mean_ScoreDice[hippocampus]"][stage1_epoch]
        acc = results.results["Mean_Accuracy"][stage1_epoch]
        print("\t".join(f"{e:.3f}" for e in chain(dice.values(), acc.values())).replace(".", ","), end="")

        # Stage 2: DICE scores and accuracy scores
        dice = results.results["Mean_ScoreDice[hippocampus]"][stage2_epoch]
        acc = results.results["Mean_Accuracy"][stage2_epoch]
        print("\t\t\t", end="")
        print("\t".join(f"{e:.3f}" for e in chain(dice.values(), acc.values())).replace(".", ","), end="")

        # Stage 3: accuracy scores
        acc = results.results["Mean_Accuracy"][stage3_epoch]
        print("\t\t\t", end="")
        print("\t".join(f"{e:.3f}" for e in acc.values()).replace(".", ","))

        print(stage1_epoch, stage2_epoch, stage3_epoch)
