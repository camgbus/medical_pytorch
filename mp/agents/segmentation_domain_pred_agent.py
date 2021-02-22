import os

import torch

from mp.agents.segmentation_agent import SegmentationAgent
from mp.data.pytorch.domain_prediction_dataset_wrapper import DomainPredictionDatasetWrapper
from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax
from mp.eval.metrics.mean_scores import get_mean_scores
from mp.utils.domain_prediction_utils import perform_stage1_training_epoch, perform_stage3_training_epoch
from mp.utils.early_stopping import EarlyStopping
from mp.utils.helper_functions import zip_longest_with_cycle
from mp.utils.pytorch.pytorch_load_restore import save_optimizer_state, load_optimizer_state


class SegmentationDomainPredictionAgent(SegmentationAgent):
    r"""An Agent for segmentation models using a classifier for the domain space using the features from the encoder"""

    def __init__(self, *args, verbose_domain_pred=False, **kwargs):
        super().__init__(*args, **kwargs)
        # Bool used to select the right outputs in self.get_outputs
        self._outputs_are_domain_predictions = False
        self.verbose_domain_pred = verbose_domain_pred

    def perform_stage1_training_epoch(self, optimizer,
                                      loss_f_classifier,
                                      loss_f_domain_pred,
                                      train_dataloaders,
                                      alpha,
                                      print_run_loss=False):
        r"""Perform a stage 1 training epoch,
        meaning that the encoder, classifier and domain predictor are all trained together

        Args:
            print_run_loss (bool): whether a running loss should be tracked and printed.
        """
        return perform_stage1_training_epoch(self, optimizer, loss_f_classifier, loss_f_domain_pred, train_dataloaders,
                                             alpha, print_run_loss=print_run_loss)

    def perform_stage2_training_epoch(self, optimizer_model,
                                      optimizer_domain_predictor,
                                      optimizer_encoder,
                                      loss_f_classifier,
                                      loss_f_domain_pred,
                                      loss_f_encoder,
                                      train_dataloaders,
                                      beta,
                                      print_run_loss=False):
        r"""Perform a stage 2 training epoch,
        meaning that the encoder, classifier and domain predictor are all trained one after the other

        Args:
            print_run_loss (bool): whether a running loss should be tracked and printed.
        """
        acc = Accumulator('loss')
        # For each batch
        for data_list in zip_longest_with_cycle(*train_dataloaders):
            classifier_losses = []
            features = []
            data_lengths = []  # Is used to produce the domain targets on the fly
            # For each dataloader
            for data in data_list:
                # Get data
                inputs, targets = self.get_inputs_targets(data)

                # Forward pass for the classification
                # Here we cannot use self.get_outputs(inputs)
                feature = self.model.get_features_from_encoder(inputs)
                outputs = softmax(self.model.get_classification_from_features(feature))

                # Store losses and predictions
                classifier_losses.append(loss_f_classifier(outputs, targets))
                features.append(feature)
                data_lengths.append(inputs.shape[0])

            # Model Optimization step
            optimizer_model.zero_grad()

            loss = torch.stack(classifier_losses, dim=0).mean()
            acc.add('loss', float(loss.detach().cpu()))

            loss.backward(retain_graph=True)
            optimizer_model.step()

            # Domain Predictor Optimization step
            optimizer_domain_predictor.zero_grad()
            features = torch.cat(features, dim=0)
            domain_pred = self.model.get_domain_prediction_from_features(features.detach())

            domain_targets = self._create_domain_targets(data_lengths)

            loss_dm = loss_f_domain_pred(domain_pred, domain_targets)
            loss_dm.backward(retain_graph=False)
            optimizer_domain_predictor.step()

            # Encoder Optimization step based on domain prediction loss
            features = []
            for data in data_list:
                # Get data
                inputs, targets = self.get_inputs_targets(data)
                feature = self.model.get_features_from_encoder(inputs)
                features.append(feature)
            features = torch.cat(features, dim=0)

            optimizer_encoder.zero_grad()
            domain_pred = self.model.get_domain_prediction_from_features(features)
            loss_encoder = beta * loss_f_encoder(domain_pred, domain_targets)
            loss_encoder.backward(retain_graph=False)
            optimizer_encoder.step()

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

    def perform_stage3_training_epoch(self,
                                      optimizer_domain_predictor,
                                      loss_f_domain_pred,
                                      train_dataloaders,
                                      print_run_loss=False):
        r"""Perform a stage 3 training epoch,
        meaning that the domain predictor only is trained

        Args:
            print_run_loss (bool): whether a running loss should be tracked and printed.
        """
        return perform_stage3_training_epoch(self, optimizer_domain_predictor, loss_f_domain_pred,
                                             train_dataloaders, print_run_loss=False)

    def train(self, results,
              optimizers,
              losses,
              train_dataloaders,
              train_dataset_names,
              init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
              eval_datasets=None, eval_interval=10,
              save_path=None, save_interval=10,
              alpha=1.0, beta=10., stage1_epochs=100):
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
        Returns:
            The the last epoch index of stages 1 to 3 as a tuple
        """
        stage1_optimizer, optimizer_model, optimizer_domain_predictor, optimizer_encoder = optimizers

        if eval_datasets is None:
            eval_datasets = dict()

        # Creating the wrappers for the datasets
        train_datasets_wrappers = {
            key: DomainPredictionDatasetWrapper(eval_datasets[key], train_dataset_names.index(key[0]))
            for key in eval_datasets if key[0] in train_dataset_names}

        loss_f_classifier, loss_f_domain_pred, loss_f_encoder = losses

        if init_epoch == 0:
            self.track_metrics(init_epoch, results, loss_f_classifier, eval_datasets)
            self.track_domain_prediction_accuracy(init_epoch, results, train_datasets_wrappers)

        # Stages 1 and 2
        for epoch in range(init_epoch, init_epoch + nr_epochs):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0 and self.verbose

            # Perform the right training step
            if epoch < stage1_epochs:
                self.perform_stage1_training_epoch(stage1_optimizer,
                                                   loss_f_classifier,
                                                   loss_f_domain_pred,
                                                   train_dataloaders,
                                                   alpha,
                                                   print_run_loss=print_run_loss)
            else:
                self.perform_stage2_training_epoch(optimizer_model,
                                                   optimizer_domain_predictor,
                                                   optimizer_encoder,
                                                   loss_f_classifier,
                                                   loss_f_domain_pred,
                                                   loss_f_encoder,
                                                   train_dataloaders,
                                                   beta,
                                                   print_run_loss=print_run_loss)

            perform_tracking = (epoch + 1) % eval_interval == 0 or \
                               epoch == stage1_epochs - 1 or \
                               epoch == init_epoch + nr_epochs - 1
            # Track statistics in results at interval or the end of stage 1 and 2
            if perform_tracking:
                self.track_metrics(epoch + 1, results, loss_f_classifier, eval_datasets)
                self.track_domain_prediction_accuracy(epoch + 1, results, train_datasets_wrappers)

            # Save agent and optimizer state at interval or the end of stage 1 and 2
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                stage1_optimizer,
                                optimizer_model,
                                optimizer_domain_predictor,
                                optimizer_encoder)

        # Stage 3
        self.track_domain_prediction_accuracy(init_epoch + nr_epochs, results, train_datasets_wrappers)
        stage3_epochs = min(20, nr_epochs)
        for epoch in range(init_epoch + nr_epochs, init_epoch + nr_epochs + stage3_epochs):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose

            self.perform_stage3_training_epoch(optimizer_domain_predictor,
                                               loss_f_domain_pred,
                                               train_dataloaders,
                                               print_run_loss=print_run_loss)

            perform_tracking = (epoch + 1) % eval_interval == 0 or epoch == stage3_epochs - 1
            # Track statistics in results
            if perform_tracking:
                self.track_metrics(epoch + 1, results, loss_f_classifier, eval_datasets)
                self.track_domain_prediction_accuracy(epoch + 1, results, train_datasets_wrappers)

            # Save agent and optimizer state
            if perform_tracking and save_path is not None:
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                stage1_optimizer,
                                optimizer_model,
                                optimizer_domain_predictor,
                                optimizer_encoder)

        return stage1_epochs - 1, nr_epochs - 1, nr_epochs - 1 + stage3_epochs

    def train_with_early_stopping(self, results, optimizers, losses, train_dataloaders, train_dataset_names,
                                  early_stopping,
                                  init_epoch=0,
                                  run_loss_print_interval=10,
                                  eval_datasets=None, eval_interval=10,
                                  save_path=None,
                                  alpha=1.0,
                                  beta=10.):
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

        loss_f_classifier, loss_f_domain_pred, loss_f_encoder = losses
        early_stopping.reset_counter()

        # Tracking metrics at step 0
        epoch = stage1_last_epoch = stage2_last_epoch = stage3_last_epoch = init_epoch
        self.track_metrics(epoch, results, loss_f_classifier, eval_datasets)
        self.track_domain_prediction_accuracy(epoch, results, train_datasets_wrappers)

        # Stage 1
        for epoch in range(epoch, 1 << 32):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0 and self.verbose

            self.perform_stage1_training_epoch(stage1_optimizer,
                                               loss_f_classifier,
                                               loss_f_domain_pred,
                                               train_dataloaders,
                                               alpha,
                                               print_run_loss=print_run_loss)
            keep_going = track_if_needed(early_stopping)
            if not keep_going:
                stage1_last_epoch = epoch + 1
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
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0 and self.verbose

            self.perform_stage2_training_epoch(optimizer_model,
                                               optimizer_domain_predictor,
                                               optimizer_encoder,
                                               loss_f_classifier,
                                               loss_f_domain_pred,
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
        # Creating a new early stopping criterion to track the accuracy of the domain predictor
        # This stage is not that important and only serves to double-check that there is
        # no more information regarding domain prediction
        early_stopping_stage3 = EarlyStopping(0, "Mean_ScoreAccuracy_DomPred",
                                              list({key[0] + "_val" for key in train_datasets_wrappers}),
                                              metric_min_delta=early_stopping.metric_min_delta)
        for epoch in range(epoch + 1, 1 << 32):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0 and self.verbose

            self.perform_stage3_training_epoch(optimizer_domain_predictor,
                                               loss_f_domain_pred,
                                               train_dataloaders,
                                               print_run_loss=print_run_loss)

            keep_going = track_if_needed(early_stopping_stage3)
            if not keep_going:
                stage3_last_epoch = epoch + 1
                if save_path is not None:
                    self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                    stage1_optimizer,
                                    optimizer_model,
                                    optimizer_domain_predictor,
                                    optimizer_encoder)
                break

        return stage1_last_epoch, stage2_last_epoch, stage3_last_epoch

    def _create_domain_targets(self, lengths):
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

        return torch.cat(domain_targets_onehot, dim=0).to(self.device)

    def track_domain_prediction_accuracy(self, epoch, results, train_datasets_wrappers):
        r"""Tracks the accuracy of the domain predictor (complementary to self.track_metrics)"""
        self._outputs_are_domain_predictions = True

        label_names = list({key[0] for key in train_datasets_wrappers})

        for ds_name, ds in train_datasets_wrappers.items():
            # Modified ds_metrics
            eval_dict = dict()
            acc = Accumulator()
            for instance_ix, instance in enumerate(ds.instances):
                subject_name = instance.name
                target = instance.y.to(self.device)
                pred = ds.predictor.get_subject_prediction(self, instance_ix)

                # Calculate metrics
                scores_dict = get_mean_scores(target, pred, metrics=["ScoreAccuracy"],
                                              label_names=label_names,
                                              label_weights=None)

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
            # End of modified ds_metrics

            for metric_key in eval_dict.keys():
                results.add(epoch=epoch, metric='Mean_' + metric_key + "_DomPred", data=ds_name,
                            value=eval_dict[metric_key]['mean'])
                results.add(epoch=epoch, metric='Std_' + metric_key + "_DomPred", data=ds_name,
                            value=eval_dict[metric_key]['std'])
            if self.verbose_domain_pred:
                print('Epoch {} dataset {}'.format(epoch, ds_name))
                for metric_key in eval_dict.keys():
                    print('{}: {}'.format(metric_key, eval_dict[metric_key]['mean']))
        self._outputs_are_domain_predictions = False

    def get_outputs(self, inputs):
        r"""Applies a softmax transformation to the model outputs"""
        # This is here so that one can still use agent.predict
        # outputs, domain_pred = self.model(inputs)
        features = self.model.get_features_from_encoder(inputs)
        if self._outputs_are_domain_predictions:
            outputs = self.model.get_domain_prediction_from_features(features)
        else:
            outputs = self.model.get_classification_from_features(features)
        return softmax(outputs)

    def save_state(self, states_path, state_name,
                   stage1_optimizer=None,
                   optimizer_model=None,
                   optimizer_domain_predictor=None,
                   optimizer_encoder=None,
                   overwrite=False):
        r"""Saves an agent state. Raises an error if the directory exists and
        overwrite=False.
        """
        # This method is overwritten because we need to save multiple optimizers
        if states_path is not None:
            # We take care of saving the optimizers ourselves
            super().save_state(states_path, state_name)
            state_full_path = os.path.join(states_path, state_name)
            for optimizer, name in ((stage1_optimizer, "stage1"),
                                    (optimizer_model, "model"),
                                    (optimizer_domain_predictor, "domain_predictor"),
                                    (optimizer_encoder, "encoder")):
                if optimizer is not None:
                    save_optimizer_state(optimizer, f'optimizer_{name}', state_full_path)

    def restore_state(self, states_path, state_name,
                      stage1_optimizer=None,
                      optimizer_model=None,
                      optimizer_domain_predictor=None,
                      optimizer_encoder=None):
        r"""Tries to restore a previous agent state, consisting of a model
        state and the content of agent_state_dict. Returns whether the restore
        operation  was successful.
        """
        # This method is overwritten because we need to save multiple optimizers
        try:
            # We take care of restoring the optimizers ourselves
            if not super().restore_state(states_path, state_name):
                return False

            if self.verbose:
                print('Trying to restore optimizer states...'.format(state_name))

            state_full_path = os.path.join(states_path, state_name)
            for optimizer, name in ((stage1_optimizer, "stage1"),
                                    (optimizer_model, "model"),
                                    (optimizer_domain_predictor, "domain_predictor"),
                                    (optimizer_encoder, "encoder")):
                if optimizer is not None:
                    load_optimizer_state(optimizer, f'optimizer_{name}', state_full_path, device=self.device)

            if self.verbose:
                print('Optimizer states {} were restored'.format(state_name))
            return True
        except:
            print('Complete state {} could not be restored'.format(state_name))
            return False
