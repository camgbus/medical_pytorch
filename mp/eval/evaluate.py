# ------------------------------------------------------------------------------
# Functions to calculate metrics and losses for subject dataloaders and datasets.
# The differences lie in that dataloaders may transform (e.g. resize) the
# targets in a way that affects the result.
# ------------------------------------------------------------------------------

import torch
from mp.eval.accumulator import Accumulator
from mp.eval.metrics.mean_scores import get_mean_scores

def dl_losses(dl, agent, loss_f):
    r"""Calculate components of the given loss for a Dataloader"""
    acc = Accumulator()
    for data in dl:
        inputs, targets = agent.get_inputs_targets(data)
        outputs = agent.get_outputs(inputs)
        # Calculate losses
        loss_dict = loss_f.get_evaluation_dict(outputs, targets)
        # Add to the accumulator
        for key, value in loss_dict.items():
            acc.add(key, value, count=len(inputs))
    return acc

def dl_metrics_subject(dl, agent, metrics, output_key=None):
    r"""Calculate metrics for a Dataloader, which is of a single subject.
    If data is 2d, join into one 3d volume.
    Note that this function only works if batch_nr=1, as is the case for the
    subject datalaoder.
    """
    acc = Accumulator()
    if len(dl) > 1:  # Slice-by-slice data
        one_channeled_target = []
        pred = []
        for data in dl:
            inputs, targets = agent.get_inputs_targets(data)
            outputs = agent.get_outputs(inputs)
            if output_key is None:
                one_channeled_target.append(agent.predict_from_outputs(targets))
                pred.append(agent.predict_from_outputs(outputs))
            else:
                one_channeled_target.append(agent.predict_from_outputs(targets)[output_key])
                pred.append(agent.predict_from_outputs(outputs)[output_key])
        one_channeled_target = torch.stack(one_channeled_target, axis=-1)
        pred = torch.stack(pred, axis=-1)
    else:
        for data in dl:
            inputs, targets = agent.get_inputs_targets(data)
            outputs = agent.get_outputs(inputs)
            one_channeled_target = agent.predict_from_outputs(targets)
            pred = agent.predict_from_outputs(outputs)
            if output_key is not None:
                one_channeled_target = one_channeled_target[output_key]
                pred = pred[output_key]
    # Calculate metrics
    scores_dict = get_mean_scores(one_channeled_target, pred, metrics=metrics,
        label_names=agent.label_names, label_weights=agent.scores_label_weights)
    # Add to the accumulator
    for key, value in scores_dict.items():
        acc.add(key, value, count=1)
    return acc

def ds_losses(ds, agent, loss_f):
    r"""Calculate components of the loss function for a Dataset.

    Args:
        ds(PytorchDataset): a PytorchDataset
        agent(Argent): an agent
        loss_f(LossAbstract): a loss function descending from LossAbstract

    Returns (dict[str -> dict]): {loss -> {subject_name -> value}}}, with 2
        additional entries per loss for 'mean' and 'std'. Note that the metric
        is calculated per dataloader per dataset. So, for instance, the scores
        for slices in a 2D dataloader are averaged.
    """
    eval_dict = dict()
    acc = Accumulator()
    for instance_ix, instance in enumerate(ds.instances):
        subject_name = instance.name
        dl = ds.get_subject_dataloader(instance_ix)
        subject_acc = dl_losses(dl, agent, loss_f)
        # Add to the accumulator and eval_dict
        for loss_key in subject_acc.get_keys():
            value = subject_acc.mean(loss_key)
            acc.add(loss_key, value, count=1)
            if loss_key not in eval_dict:
                eval_dict[loss_key] = dict()
            eval_dict[loss_key][subject_name] = value
    # Add mean and std values to the eval_dict
    for loss_key in acc.get_keys():
        eval_dict[loss_key]['mean'] = acc.mean(loss_key)
        eval_dict[loss_key]['std'] = acc.std(loss_key)
    return eval_dict

def ds_metrics(ds, agent, metrics):
    r"""Calculate metrics for a Dataset.

    Args:
        ds(PytorchDataset): a PytorchDataset
        agent(Argent): an agent
        metrics(list[str]): a list of metric names

    Returns (dict[str -> dict]): {metric -> {subject_name -> value}}}, with 2
        additional entries per metric for 'mean' and 'std'.
    """
    eval_dict = dict()
    acc = Accumulator()
    for instance_ix, instance in enumerate(ds.instances):
        subject_name = instance.name

        target = instance.y.tensor.to(agent.device)
        pred = ds.predictor.get_subject_prediction(agent, instance_ix)

        # Calculate metrics
        scores_dict = get_mean_scores(target, pred, metrics=metrics,
                    label_names=agent.label_names,
                    label_weights=agent.scores_label_weights)
        # Add to the accumulator and eval_dict
        for metric_key, value in scores_dict.items():
            acc.add(metric_key, value, count=1)
            if metric_key not in eval_dict:
                eval_dict[metric_key] = dict()
            eval_dict[metric_key][subject_name] = value
    # Add mean and std values to the eval_dict
    for metric_key in acc.get_keys():
        eval_dict[metric_key]['mean'] = acc.mean(metric_key)
        eval_dict[metric_key]['std'] = acc.std(metric_key)
    return eval_dict

def ds_metrics_multiple(ds, agent, metrics):
    r"""Variant of ds_metrics when there are several inputs and outputs.
    Here, the dataloader is used, so the original size is not restored, but a 
    3d volume is reconstructed when necessary.
    """
    assert isinstance(metrics, dict)
    eval_dict = dict()
    acc = Accumulator()
    for instance_ix, instance in enumerate(ds.instances):
        subject_name = instance.name
        dl = ds.get_subject_dataloader(instance_ix)
        for output_key, key_metrics in metrics.items():
            subject_acc = dl_metrics_subject(dl, agent, key_metrics, output_key)
            # Add to the accumulator and eval_dict
            for metric_key in subject_acc.get_keys():
                value = subject_acc.mean(metric_key)
                full_metric_key = str(output_key) + '_' + str(metric_key)
                acc.add(full_metric_key, value, count=1)
                if full_metric_key not in eval_dict:
                    eval_dict[full_metric_key] = dict()
                eval_dict[full_metric_key][subject_name] = value
    # Add mean and std values to the eval_dict
    for metric_key in acc.get_keys():
        eval_dict[metric_key]['mean'] = acc.mean(metric_key)
        eval_dict[metric_key]['std'] = acc.std(metric_key)
    return eval_dict

def ds_losses_metrics(ds, agent, loss_f, metrics, multiple=False):
    r"""Combination of metrics and losses into one dictionary."""
    eval_dict = ds_losses(ds, agent, loss_f)
    if metrics:
        if multiple:
            eval_dict.update(ds_metrics_multiple(ds, agent, metrics))
        else:
            eval_dict.update(ds_metrics(ds, agent, metrics))
    return eval_dict
