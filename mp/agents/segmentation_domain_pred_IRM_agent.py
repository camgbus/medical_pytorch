import torch

from mp.agents.segmentation_domain_pred_agent import SegmentationDomainAgent
from mp.data.pytorch.domain_prediction_dataset_wrapper import DomainPredictionDatasetWrapper
from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax
from mp.eval.losses.losses_irm import IRMLossAbstract
from mp.utils.early_stopping import EarlyStopping
from mp.utils.helper_functions import zip_longest_with_cycle


class SegmentationDomainIRMAgent(SegmentationDomainAgent):
    r"""
    An Agent for segmentation models using a classifier for the domain space using the features from the encoder.
    Uses IRM training schedule for the domain predictor head during stages 1 and 3.
    """

    def perform_stage1_training_epoch(self, optimizer,
                                      loss_f_classifier,
                                      irm_loss_f_domain_pred,
                                      train_dataloaders,
                                      print_run_loss=False):
        r"""Perform a stage 1 training epoch,
        meaning that the encoder, classifier and domain predictor are all trained together

        Args:
            irm_loss_f_domain_pred (IRMLossAbstract): the IRM loss function for the domain predictor
            print_run_loss (bool): whether a running loss should be tracked and printed.
        """
        acc = Accumulator('loss')
        dl_cnt = len(train_dataloaders)
        # For each batch
        for data_list in zip_longest_with_cycle(*train_dataloaders):
            classifier_losses = []
            domain_pred_losses = []
            domain_pred_penalties = []
            # For each dataloader
            for idx, data in enumerate(data_list):
                # Get data
                inputs, targets = self.get_inputs_targets(data)

                # Forward pass for the classification and domain prediction
                # Here we cannot use self.get_outputs(inputs)
                classifier_output, domain_pred = self.model(inputs, detach=True)

                # Store losses and predictions
                classifier_losses.append(loss_f_classifier(softmax(classifier_output), targets))

                # We create the domain targets
                domain_targets = torch.zeros((inputs.shape[0], dl_cnt), dtype=targets.dtype).to(self.device)
                domain_targets[:, idx] = 1
                domain_pred_output = softmax(domain_pred)

                # Computing ERM and IRM terms for domain prediction
                domain_pred_losses.append(irm_loss_f_domain_pred.erm(domain_pred_output, domain_targets))
                domain_pred_penalties.append(irm_loss_f_domain_pred(domain_pred_output, domain_targets))

            # Optimization step
            optimizer.zero_grad()
            loss = torch.stack(classifier_losses, dim=0).mean() + \
                   irm_loss_f_domain_pred.finalize_loss(domain_pred_losses, domain_pred_penalties)

            loss.backward()
            optimizer.step()
            acc.add('loss', float(loss.detach().cpu()))

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

    def perform_stage3_training_epoch(self,
                                      optimizer_domain_predictor,
                                      irm_loss_f_domain_pred,
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
            domain_pred_losses = []
            domain_pred_penalties = []
            # For each dataloader
            for idx, data in enumerate(data_list):
                # Get data
                inputs, targets = self.get_inputs_targets(data)

                # Forward pass for the domain prediction
                # Here we cannot use self.get_outputs(inputs)
                feature = self.model.get_features_from_encoder(inputs).detach()
                domain_pred = softmax(self.model.get_domain_prediction_from_features(feature))

                # We create the domain targets
                domain_targets = torch.zeros((inputs.shape[0], dl_cnt), dtype=targets.dtype).to(self.device)
                domain_targets[:, idx] = 1
                domain_pred_output = softmax(domain_pred)

                # Computing ERM and IRM terms for domain prediction
                domain_pred_losses.append(irm_loss_f_domain_pred.erm(domain_pred_output, domain_targets))
                domain_pred_penalties.append(irm_loss_f_domain_pred(domain_pred_output, domain_targets))

            # Domain Predictor Optimization step
            optimizer_domain_predictor.zero_grad()
            loss_dm = irm_loss_f_domain_pred.finalize_loss(domain_pred_losses, domain_pred_penalties)
            loss_dm.backward()
            optimizer_domain_predictor.step()
            acc.add("loss", float(loss_dm.detach().cpu()))

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

    def train(self, results,
              optimizers,
              losses,
              train_dataloaders,
              init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
              eval_datasets=None, eval_interval=10,
              save_path=None, save_interval=10,
              beta=10., penalty_weight=1e5, stage1_epochs=100):
        # TODO maybe implement this later
        # The reason why this is not implemented is that the user would need to give a few more hyper-parameters
        # in the form of how many epochs should each stage of IRM training last, given that there are 2 IRM trainings
        # within the 3 stages of the domain prediction training
        raise NotImplementedError

    def train_with_early_stopping(self, results, optimizers, losses, train_dataloaders, train_dataset_names,
                                  early_stopping,
                                  init_epoch=0,
                                  run_loss_print_interval=10,
                                  eval_datasets=None, eval_interval=10,
                                  save_path=None,
                                  beta=10., penalty_weight=1e5):
        r"""Train a model through its agent. Performs training epochs,
        tracks metrics and saves model states.

        Args:
            optimizers (list): a list containing the 4 following optimizers (in order):
                                    - one for all of the models parameters
                                    - one for the model's parameters (encoder + classifier)
                                    - one for the domain predictor only
                                    - one for the encoder only
            losses (list): a list containing the following losses (in order):
                                    - one for the classifier
                                    - one for the domain predictor
                                    - one for the encoder (based on the domain predictions)
            train_dataloaders (list): a list of Dataloader
            train_dataset_names (list): the list of the names of the dataset used for training
                                        (same order as for the train_dataloaders)
            early_stopping (EarlyStopping): the early stopping criterion
        Returns:
            The the last epoch index of stages 1 to 3 as a tuple
        """

        def track_if_needed(early_stopping_criterion):
            # Track statistics in results object at interval and returns whether training should keep going
            if (epoch + 1) % eval_interval == 0:
                self.track_metrics(epoch + 1, results, loss_f_classifier, eval_datasets)
                self.track_domain_prediction_accuracy(epoch + 1, results, train_datasets_wrappers)
                return early_stopping_criterion.check_results(results, epoch + 1)

            return True

        stage1_optimizer, optimizer_model, optimizer_domain_predictor, optimizer_encoder = optimizers

        if eval_datasets is None:
            eval_datasets = dict()

        # Creating the wrappers for the datasets
        train_datasets_wrappers = {
            key: DomainPredictionDatasetWrapper(eval_datasets[key], train_dataset_names.index(key[0]))
            for key in eval_datasets if key[0] in train_dataset_names}

        loss_f_classifier, irm_loss_f_domain_pred, loss_f_encoder = losses
        early_stopping.reset()
        early_stopping_domain_pred = EarlyStopping(0, "Mean_ScoreAccuracy_DomPred",
                                                   list({key[0] + "_val" for key in train_datasets_wrappers}),
                                                   metric_min_delta=early_stopping.metric_min_delta)
        # This penalty weight attribute needs to get reset between runs
        irm_loss_f_domain_pred.penalty_weight = 1.

        epoch = stage2_last_epoch = init_epoch
        stage1_last_epochs = [init_epoch, init_epoch]
        stage3_last_epochs = [init_epoch, init_epoch]

        # Tracking metrics
        self.track_metrics(epoch, results, loss_f_classifier, eval_datasets)
        self.track_domain_prediction_accuracy(epoch, results, train_datasets_wrappers)

        # Stage 1 Domain Prediction
        # Sub-stage 1 IRM
        for epoch in range(epoch, 1 << 32):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0 and self.verbose

            self.perform_stage1_training_epoch(stage1_optimizer,
                                               loss_f_classifier,
                                               irm_loss_f_domain_pred,
                                               train_dataloaders,
                                               print_run_loss=print_run_loss)
            keep_going = track_if_needed(early_stopping)
            if not keep_going:
                stage1_last_epochs[0] = epoch + 1
                if save_path is not None:
                    self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                    stage1_optimizer,
                                    optimizer_model,
                                    optimizer_domain_predictor,
                                    optimizer_encoder)
                break

        # Sub-stage 2 IRM, but only training domain prediction
        irm_loss_f_domain_pred.penalty_weight = penalty_weight
        early_stopping_domain_pred.check_results(results, epoch + 1)  # Init the best scores for this criterion
        for epoch in range(epoch + 1, 1 << 32):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0 and self.verbose

            self.perform_stage3_training_epoch(optimizer_domain_predictor,
                                               irm_loss_f_domain_pred,
                                               train_dataloaders,
                                               print_run_loss=print_run_loss)
            keep_going = track_if_needed(early_stopping_domain_pred)
            if not keep_going:
                stage1_last_epochs[1] = epoch + 1
                if save_path is not None:
                    self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                    stage1_optimizer,
                                    optimizer_model,
                                    optimizer_domain_predictor,
                                    optimizer_encoder)
                break

        # Stage 2
        early_stopping.reset_counter()
        for epoch in range(epoch + 1, 1 << 32):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose

            # The only difference is that we use the ERM loss instead of the IRM one
            self.perform_stage2_training_epoch(optimizer_model,
                                               optimizer_domain_predictor,
                                               optimizer_encoder,
                                               loss_f_classifier,
                                               irm_loss_f_domain_pred.erm_loss,
                                               loss_f_encoder,
                                               train_dataloaders,
                                               beta,
                                               print_run_loss=print_run_loss)
            keep_going = track_if_needed(early_stopping)
            if not keep_going:
                stage2_last_epoch = epoch + 1
                if save_path is not None:
                    self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                    stage1_optimizer,
                                    optimizer_model,
                                    optimizer_domain_predictor,
                                    optimizer_encoder)
                break

        # Stage 3
        # This stage is not that important and only serves to double-check that there is
        # no more information regarding domain prediction
        early_stopping_domain_pred.reset()
        irm_loss_f_domain_pred.penalty_weight = 1.

        # Sub-stages 1 & 2 IRM
        for sub_stage, criterion in enumerate([early_stopping, early_stopping_domain_pred]):
            for epoch in range(epoch + 1, 1 << 32):
                print_run_loss = (epoch + 1) % run_loss_print_interval == 0 and self.verbose

                self.perform_stage3_training_epoch(optimizer_domain_predictor,
                                                   irm_loss_f_domain_pred,
                                                   train_dataloaders,
                                                   print_run_loss=print_run_loss)
                keep_going = track_if_needed(early_stopping_domain_pred)
                if not keep_going:
                    stage3_last_epochs[sub_stage] = epoch + 1
                    if save_path is not None:
                        self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                        stage1_optimizer,
                                        optimizer_model,
                                        optimizer_domain_predictor,
                                        optimizer_encoder)
                    break

            # Update penalty loss for next IRM sub-stage
            irm_loss_f_domain_pred.penalty_weight = penalty_weight
            # Also reset the early stopping criterion
            # This is a reset instead of reset_counter call, because a drop in performance is expected her
            # And we have no guarantee that we will reach our max again within the patience timeframe
            # So we want to train until there's no more improvement (same as in SegmentationDomainAgent)
            early_stopping_domain_pred.reset()

        return tuple(stage1_last_epochs), stage2_last_epoch, tuple(stage3_last_epochs)
