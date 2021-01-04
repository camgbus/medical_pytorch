# ------------------------------------------------------------------------------
# The same code as in example_training.ipynp but as a python module instead of 
# jupyter notebook. See that file for more detailed explanations.
# ------------------------------------------------------------------------------

# 1. Imports

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
from mp.models.segmentation.unet_fepegar import UNet3D
from mp.models.IRMGamesModel import IRMGamesModel


# 2. Define configuration
config = {'experiment_name': 'test_exp_not_decath4_firm_aug_many', 'device': 'cuda:0',
          'nr_runs': 1, 'cross_validation': False, 'val_ratio': 0.0, 'test_ratio': 0.3,
          'input_shape': (1, 48, 64, 64), 'resize': False, 'augmentation': 'geometric',
          'class_weights': (0., 1.), 'lr': 2e-4, 'batch_size': 22,
          "nr_epochs": 100,
          "eval_interval": 10
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
    ds_lengths = [len(datasets[name, "train"]) for name in train_ds_names]
    total_length = sum(ds_lengths)
    dls = [DataLoader(datasets[name, "train"], batch_size=config['batch_size'] * length // total_length, shuffle=True)
           for name, length in zip(train_ds_names, ds_lengths)]

    # 8. Initialize model
    model = IRMGamesModel([UNet3D(input_shape, nr_labels) for _ in train_ds_names],
                          input_shape=input_shape,
                          output_shape=UNet3D(input_shape, nr_labels).output_shape,
                          # representation_learner=UNet3D(input_shape, input_shape[0])
                          )
    model.to(device)

    # 9. Define loss and optimizer
    loss_g = LossDiceBCE(bce_weight=1., smooth=1., device=device)
    loss_f = LossClassWeighted(loss=loss_g, weights=config['class_weights'],
                               device=device)
    optimizers = [optim.Adam(sub_model.parameters(), lr=config['lr']) for sub_model in model.models]

    # 10. Train model
    results = Result(name='training_trajectory')
    agent = SegmentationIRMGamesAgent(model=model, label_names=label_names, device=device)
    agent.train(results, optimizers, loss_g, train_dataloaders=dls,
                init_epoch=0, nr_epochs=config["nr_epochs"], run_loss_print_interval=config["eval_interval"],
                eval_datasets=datasets, eval_interval=config["eval_interval"],
                save_path=exp_run.paths['states'], save_interval=config["nr_epochs"])

    # 11. Save and print results for this experiment run
    exp_run.finish(results=results, plot_metrics=['Mean_ScoreDice[hippocampus]'])

    metric = results.results["Mean_ScoreDice[hippocampus]"][max(results.results["Mean_ScoreDice[hippocampus]"])]
    # Print the mean dice score for each dataset and train/test subset
    for key in metric:
        print(f"{key}: {metric[key]:.3f}")
    # Print the final result in a csv format for easy copy-pasting to the spreadsheet
    print("Finished: " + "\t".join(f"{e:.3f}" for e in metric.values()).replace(".", ","))
