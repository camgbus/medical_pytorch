import torch

from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax
from mp.eval.losses.losses_irm import IRMLossAbstract
from mp.utils.helper_functions import zip_longest_with_cycle


def _create_domain_targets(lengths, device):
    r"""
    Builds the tensors containing the domain targets. We only need to do this once as there will
    always be the same number of subjects per domain per batch.
    Args:
        lengths (list): a list of lengths
    """
    # There will always be the same number of subjects per domain per batch
    # So we don't need to store the domain in each datasets and we only need to create the tensors once for training
    nb_domains = len(lengths)
    domain_targets_onehot = list()
    # We need to find out the batch size for each domain
    for idx, length in enumerate(lengths):
        domain_target = torch.zeros((length, nb_domains), dtype=torch.float)
        domain_target[:, idx] = 1.
        domain_targets_onehot.append(domain_target)

    return torch.cat(domain_targets_onehot, dim=0).to(device)


def perform_stage1_training_epoch(agent,
                                  optimizer,
                                  loss_f_classifier,
                                  loss_f_domain_pred,
                                  train_dataloaders,
                                  alpha,
                                  print_run_loss=False):
    """
    Performs the first stage of Domain Prediction training.
    Automatically detects whether the loss for the classifier or the domain predictor is IRM
    """
    acc = Accumulator('loss')
    dl_cnt = len(train_dataloaders)  # How many training environments

    # For each batch
    for data_list in zip_longest_with_cycle(*train_dataloaders):
        # Classifier ERM and IRM vars
        classifier_losses = []
        classifier_penalties = []
        # Domain prediction IRM vars
        domain_pred_losses = []
        domain_pred_penalties = []
        # Domain prediction ERM vars
        domain_preds = []
        data_lengths = []  # Is used to produce the domain targets on the fly

        # For each dataloader
        for idx, data in enumerate(data_list):  # training env ix, data from corresponding dataloader
            # Get data
            inputs, targets = agent.get_inputs_targets(data)

            # Forward pass for the classification and domain prediction
            classifier_output, domain_pred = agent.model(inputs)

            # Computing ERM terms for classification (and IRM terms if needed)
            classifier_output = softmax(classifier_output)
            if isinstance(loss_f_classifier, IRMLossAbstract):
                classifier_losses.append(loss_f_classifier.erm(classifier_output, targets))
                classifier_penalties.append(loss_f_classifier(classifier_output, targets))
            else:
                classifier_losses.append(loss_f_classifier(classifier_output, targets))

            # Computing ERM terms for Domain Prediction (and IRM terms if needed)
            if isinstance(loss_f_domain_pred, IRMLossAbstract):
                # We create the domain targets
                domain_targets = torch.zeros((inputs.shape[0], dl_cnt), dtype=targets.dtype).to(agent.device)
                domain_targets[:, idx] = 1
                domain_pred_output = softmax(domain_pred)

                domain_pred_losses.append(loss_f_domain_pred.erm(domain_pred_output, domain_targets))
                domain_pred_penalties.append(loss_f_domain_pred(domain_pred_output, domain_targets))
            else:
                domain_preds.append(domain_pred)
                data_lengths.append(inputs.shape[0])

        # Optimization step
        optimizer.zero_grad()

        # Classifier
        if isinstance(loss_f_classifier, IRMLossAbstract):
            classifier_loss = loss_f_classifier.finalize_loss(classifier_losses, classifier_penalties)
        else:
            classifier_loss = torch.stack(classifier_losses, dim=0).mean()

        # Domain predictor
        if isinstance(loss_f_domain_pred, IRMLossAbstract):
            domain_predictor_loss = loss_f_domain_pred.finalize_loss(domain_pred_losses, domain_pred_penalties)
        else:
            domain_preds = softmax(torch.cat(domain_preds, dim=0))
            domain_targets = _create_domain_targets(data_lengths, agent.device)
            domain_predictor_loss = loss_f_domain_pred(domain_preds, domain_targets)

        # Overall loss
        loss = classifier_loss + alpha * domain_predictor_loss

        loss.backward()
        optimizer.step()
        acc.add('loss', float(loss.detach().cpu()))

    if print_run_loss:
        print('\nRunning loss: {}'.format(acc.mean('loss')))


def perform_stage3_training_epoch(agent,
                                  optimizer_domain_predictor,
                                  loss_f_domain_pred,
                                  train_dataloaders,
                                  print_run_loss=False):
    r"""Perform a stage 3 training epoch,
    meaning that the domain predictor only is trained

    Args:
        irm_loss_f_domain_pred (IRMLossAbstract): the IRM loss function for the domain predictor
        print_run_loss (bool): whether a running loss should be tracked and printed.
    """
    acc = Accumulator('loss')
    dl_cnt = len(train_dataloaders)
    # For each batch
    for data_list in zip_longest_with_cycle(*train_dataloaders):
        # Domain prediction IRM vars
        domain_pred_losses = []
        domain_pred_penalties = []
        # Domain prediction ERM vars
        domain_preds = []
        data_lengths = []  # Is used to produce the domain targets on the fly
        # For each dataloader
        for idx, data in enumerate(data_list):  # training env ix, data from corresponding dataloader
            # Get data
            inputs, targets = agent.get_inputs_targets(data)

            # Forward pass for the domain prediction
            # Here we cannot use self.get_outputs(inputs)
            feature = agent.model.get_features_from_encoder(inputs).detach()
            domain_pred = softmax(agent.model.get_domain_prediction_from_features(feature))

            # Computing ERM terms for Domain Prediction (and IRM terms if needed)
            if isinstance(loss_f_domain_pred, IRMLossAbstract):
                # We create the domain targets
                domain_targets = torch.zeros((inputs.shape[0], dl_cnt), dtype=targets.dtype).to(agent.device)
                domain_targets[:, idx] = 1
                domain_pred_output = softmax(domain_pred)

                domain_pred_losses.append(loss_f_domain_pred.erm(domain_pred_output, domain_targets))
                domain_pred_penalties.append(loss_f_domain_pred(domain_pred_output, domain_targets))
            else:
                domain_preds.append(domain_pred)
                data_lengths.append(inputs.shape[0])


        # Domain Predictor Optimization step
        optimizer_domain_predictor.zero_grad()

        # Domain predictor
        if isinstance(loss_f_domain_pred, IRMLossAbstract):
            domain_predictor_loss = loss_f_domain_pred.finalize_loss(domain_pred_losses, domain_pred_penalties)
        else:
            domain_preds = softmax(torch.cat(domain_preds, dim=0))
            domain_targets = _create_domain_targets(data_lengths, agent.device)
            domain_predictor_loss = loss_f_domain_pred(domain_preds, domain_targets)

        domain_predictor_loss.backward()
        optimizer_domain_predictor.step()
        acc.add("loss", float(domain_predictor_loss.detach().cpu()))

    if print_run_loss:
        print('\nRunning loss: {}'.format(acc.mean('loss')))
