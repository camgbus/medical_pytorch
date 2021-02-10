## A *quick* Q&A regarding scripts

**Q1.** What is this multitude of scripts named `example_training_*.py` and what do they do?

**A.** These are copy-pastes from `example_training.py`, but they use different agents for different purposes:
- `example_training_irm.py`: trains a model using the IRM training schedule (http://arxiv.org/abs/1907.02893)
  (can also be used to train a model on multiple datasets with an ERM loss)
- `example_training_irm_games.py`: trains an ensemble of model as described to the IRM Games paper
  (http://arxiv.org/abs/2002.04692)
- `example_training_domain_pred.py`: trains a model with **Domain Prediction** as described in the papers
  written by N. K. Dinsdale (doi: 10.1101/2020.10.09.332973)
- `example_training_domain_pred_irm_dp.py`: trains a model with **Domain Prediction** but an IRM loss
  is used on the domain predictor
- `example_training_domain_pred_irm_seg.py`: trains a model with **Domain Prediction** but an IRM loss
  is used on the segmentor
 - `example_training_domain_pred_irm_both.py`: trains a model with **Domain Prediction** but an IRM loss
 is used on both the segmentor and the domain predictor

**Q2.** Are there any other changes?

**A.** Yes, some steps in the script have been reordered. Most parameters can be configured in a
  config dictionary, but the script can store a list of such dictionaries and run the experiments sequentially.
  Of course, there some new keys for these dictionaries but more about this later. Lastly, these scripts
  use an automatic stopping criterion for training rather than a fixed number of epochs. As such,
  the epoch at which a training stage ends is stored in `epochs.pkl` in the run's `obj` folder.
  They are also marked in the plot using a red vertical line.

**Q3.** Cool, cool, but how do I use these automatic stopping criteria? 

**A.** An `EarlyStopping` object requires a patience (an int) and a list of split names to monitor.
  If your dataset in named `dataset` and you want to monitor the validation split, than the split name
  will be `dataset_val`. If you to monitor another split, you can replace `val` by `train` or `test`.
  The object will then look at a specified metric (DICE score in most agents) and check whether there has been
  a sufficiently large improvement on the maximum reached for any of the tracked splits.
  
## Common parameters between scripts

- `experiment_name`: the experiment name, it will also be the name of the folder in which the experiment is stored
- `device`: the device on which to train for instance `cuda:0`
- `nr_runs`: the number of runs (an integer)
- `cross_validation`: whether a `nr_runs`-fold cross-validation should be performed
- `val_ratio`: the validation ratio (a float)
- `test_ratio`: the test ratio (a float), is ignored if `cross_validation` is True
- `input_shape`: the input shape for the model (a tuple) 
- `resize`: whether the input should be resized to the `input_shape`, otherwise it will be zero_padded or center cropped
- `augmentation`: the augmentation strategy (a string)
- `class_weights`: the class weights for the class-weighted loss (a tuple)
- `lr`: the learning rate (a float)
- `eval_interval`: the epoch interval at which the metrics are computed for all splits 
  (and at which the `EarlyStopping` object is updated)
- `train_ds_names`: the list of dataset names the model should train on
- `batch_sizes`: a list of batch sizes corresponding one-to-one with the list of training dataset names

## Script-specific parameters

Any parameter not listed above and found in the example config for a script is script-specific,
but there only a couple of them.

For scripts containing `_irm` in their name, `penalty_weight` corresponds to the lambda balancing between the ERM 
and IRM terms in the paper. The IRM loss function can be selected using the `loss` parameter and can take the following 
values: `"erm"`, `"irmv1"`, `"vrex"`, `"mmrex"`.

For scripts containing `_domain_pred` in their name, `beta` is the scalar in front of the encoder confusion-loss term.

## Script-specific model wrappers

The scripts for IRM Games and Domain Prediction require the base model to be wrapped.

In the case of IRM Games, we are training an ensemble of models using a `IRMGamesModel` object as a wrapper.
It takes as input a list of models (same length as the number of training datasets).

Scripts doing **Domain Prediction** require the model's class to inherit from the interface `FullModelWithDomainPred`.
This allows to access each a of the parts of the architecture (encoder, segmentor and domain predictor) individually.
An example of such a class can be found in `mp/models/domain_prediction/unet_with_domain_pred.py`.
