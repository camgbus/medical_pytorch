import os

import torch

from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax
from mp.utils.helper_functions import zip_longest_with_cycle
from mp.utils.pytorch.pytorch_load_restore import save_optimizer_state, load_optimizer_state
from mp.utils.early_stopping import EarlyStopping


class SegmentationDomainAgent(SegmentationAgent):
    r"""An Agent for segmentation models using a classifier for the domain space using the features from the encoder"""

    def perform_stage1_training_epoch(self, optimizer,
                                      loss_f_classifier,
                                      loss_f_domain_pred,
                                      train_dataloaders,
                                      print_run_loss=False):
        r"""Perform a stage 1 training epoch,
        meaning that the encoder, classifier and domain predictor are all trained together

        Args:
            print_run_loss (bool): whether a running loss should be tracked and printed.
        """
        acc = Accumulator('loss')
        # For each batch
        for data_list in zip_longest_with_cycle(*train_dataloaders):
            classifier_losses = []
            domain_preds = []
            data_lengths = []  # Is used to produce the domain targets on the fly
            # For each dataloader
            for data in data_list:
                # Get data
                inputs, targets = self.get_inputs_targets(data)

                # Forward pass for the classification and domain prediction
                # Here we cannot use self.get_outputs(inputs)
                classifier_output, domain_pred = self.model(inputs, detach=True)

                # Store losses and predictions
                classifier_losses.append(loss_f_classifier(softmax(classifier_output), targets))
                domain_preds.append(domain_pred)
                data_lengths.append(inputs.shape[0])

            # Domain prediction
            domain_preds = softmax(torch.cat(domain_preds, dim=0))
            domain_targets = self._create_domain_targets(data_lengths)

            # Optimization step
            optimizer.zero_grad()
            loss = torch.stack(classifier_losses, dim=0).mean() + loss_f_domain_pred(domain_preds, domain_targets)

            loss.backward()
            optimizer.step()
            acc.add('loss', float(loss.detach().cpu()))

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

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
        acc = Accumulator('loss')
        # For each batch
        for data_list in zip_longest_with_cycle(*train_dataloaders):
            domain_preds = []
            data_lengths = []  # Is used to produce the domain targets on the fly
            # For each dataloader
            for data in data_list:
                # Get data
                inputs, targets = self.get_inputs_targets(data)

                # Forward pass for the domain prediction
                # Here we cannot use self.get_outputs(inputs)
                feature = self.model.get_features_from_encoder(inputs).detach()
                domain_pred = softmax(self.model.get_domain_prediction_from_features(feature))

                # Store losses and predictions
                domain_preds.append(domain_pred)
                data_lengths.append(inputs.shape[0])

            # Domain Predictor Optimization step
            optimizer_domain_predictor.zero_grad()
            domain_preds = torch.cat(domain_preds, dim=0)

            domain_targets = self._create_domain_targets(data_lengths)

            loss_dm = loss_f_domain_pred(domain_preds, domain_targets)
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
              beta=10., stage1_epochs=100):
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
        Returns:
            The the last epoch index of stages 1 to 3 as a tuple
        """
        stage1_optimizer, optimizer_model, optimizer_domain_predictor, optimizer_encoder = optimizers

        if eval_datasets is None:
            eval_datasets = dict()

        loss_f_classifier, loss_f_domain_pred, loss_f_encoder = losses

        if init_epoch == 0:
            self.track_metrics(init_epoch, results, loss_f_classifier, eval_datasets)
            self.track_domain_prediction_accuracy(init_epoch, results, train_dataloaders)

        # Stages 1 and 2
        for epoch in range(init_epoch, init_epoch + nr_epochs):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0 and self.verbose

            # Perform the right training step
            if epoch < stage1_epochs:
                self.perform_stage1_training_epoch(stage1_optimizer,
                                                   loss_f_classifier,
                                                   loss_f_domain_pred,
                                                   train_dataloaders,
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
                self.track_domain_prediction_accuracy(epoch + 1, results, train_dataloaders)

            # Save agent and optimizer state at interval or the end of stage 1 and 2
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                stage1_optimizer,
                                optimizer_model,
                                optimizer_domain_predictor,
                                optimizer_encoder)

        # Stage 3
        self.track_domain_prediction_accuracy(init_epoch, results, train_dataloaders)
        stage3_epochs = 20
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
                self.track_domain_prediction_accuracy(epoch + 1, results, train_dataloaders)

            # Save agent and optimizer state
            if perform_tracking and save_path is not None:
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                stage1_optimizer,
                                optimizer_model,
                                optimizer_domain_predictor,
                                optimizer_encoder)

        return stage1_epochs - 1, nr_epochs - 1, nr_epochs - 1 + stage3_epochs


    def train_with_early_stopping(self, results, optimizers, losses, train_dataloaders, early_stopping,
                                  init_epoch=0,
                                  run_loss_print_interval=10,
                                  eval_datasets=None, eval_interval=10,
                                  save_path=None,
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
            early_stopping (EarlyStopping): the early stopping criterion
        Returns:
            The the last epoch index of stages 1 to 3 as a tuple
        """

        def track_if_needed(early_stopping_criterion):
            # Track statistics in results object at interval and returns whether training should keep going
            if (epoch + 1) % eval_interval == 0:
                self.track_metrics(epoch + 1, results, loss_f_classifier, eval_datasets)
                self.track_domain_prediction_accuracy(epoch + 1, results, train_dataloaders)
                return early_stopping_criterion.check_results(results, epoch + 1)

            return True

        stage1_optimizer, optimizer_model, optimizer_domain_predictor, optimizer_encoder = optimizers

        if eval_datasets is None:
            eval_datasets = dict()

        loss_f_classifier, loss_f_domain_pred, loss_f_encoder = losses
        early_stopping.reset_counter()

        # Tracking metrics at step 0
        epoch = stage1_last_epoch = stage2_last_epoch = stage3_last_epoch = init_epoch
        self.track_metrics(epoch, results, loss_f_classifier, eval_datasets)
        self.track_domain_prediction_accuracy(epoch, results, train_dataloaders)

        # Stage 1
        for epoch in range(epoch, 1 << 32):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0 and self.verbose

            self.perform_stage1_training_epoch(stage1_optimizer,
                                               loss_f_classifier,
                                               loss_f_domain_pred,
                                               train_dataloaders,
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
        # FIXME Currently using an early stopping criterion will produce bad results:
        #       The domain predictor will constantly predict the larger dataset
        # early_stopping_stage3 = EarlyStopping(0, "Mean_Accuracy", ["Training dl 0", "Training dl 1"],
        #                                       metric_min_delta=early_stopping.metric_min_delta)
        # for epoch in range(epoch + 1, 1 << 32):
        for epoch in range(epoch + 1, epoch + 21):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0 and self.verbose

            self.perform_stage3_training_epoch(optimizer_domain_predictor,
                                               loss_f_domain_pred,
                                               train_dataloaders,
                                               print_run_loss=print_run_loss)

            if (epoch + 1) % eval_interval == 0:
                self.track_metrics(epoch + 1, results, loss_f_classifier, eval_datasets)
                self.track_domain_prediction_accuracy(epoch + 1, results, train_dataloaders)

            # keep_going = track_if_needed(early_stopping_stage3)
            # if not keep_going:
            #     stage3_last_epoch = epoch + 1
            #     if save_path is not None:
            #         self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
            #                         stage1_optimizer,
            #                         optimizer_model,
            #                         optimizer_domain_predictor,
            #                         optimizer_encoder)
            #     break

        stage3_last_epoch = epoch + 1
        if save_path is not None:
            self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                            stage1_optimizer,
                            optimizer_model,
                            optimizer_domain_predictor,
                            optimizer_encoder)

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

    def track_domain_prediction_accuracy(self, epoch, results, train_dataloaders):
        r"""Tracks the accuracy of the domain predictor (complementary to self.track_metrics)"""
        if self.verbose:
            print('Epoch {} accuracy domain prediction'.format(epoch))

        # Accumulator for the accuracy across datasets
        overall_acc = Accumulator()
        dl_cnt = len(train_dataloaders)

        # For each dataloader
        for idx, dl in enumerate(train_dataloaders):
            acc = Accumulator()
            # For each batch
            for data in dl:
                inputs, targets = self.get_inputs_targets(data)

                # We create the domain targets
                domain_targets = torch.zeros((inputs.shape[0], dl_cnt), dtype=targets.dtype).to(self.device)
                domain_targets[:, idx] = 1

                one_channeled_target = self.predict_from_outputs(domain_targets)

                # Predicting...
                feature = self.model.get_features_from_encoder(inputs)
                domain_pred = self.predict_from_outputs(self.model.get_domain_prediction_from_features(feature))

                # Computing the accuracy
                # noinspection PyTypeChecker
                score = torch.sum(domain_pred == one_channeled_target).cpu().numpy() / inputs.shape[0]
                acc.add("Accuracy", score, inputs.shape[0])
                overall_acc.add("Accuracy", score, inputs.shape[0])

            # Add to the accumulator
            results.add(epoch=epoch, metric='Mean_Accuracy', data=f"Training dl {idx}", value=acc.mean("Accuracy"))

            if self.verbose:
                print(f"ScoreAccuracy[Dataloader {idx}]: {acc.mean('Accuracy')}")

        if self.verbose:
            print(f"ScoreAccuracy: {overall_acc.mean('Accuracy')}")

    def get_outputs(self, inputs):
        r"""Applies a softmax transformation to the model outputs"""
        # This is here so that one can still use agent.predict
        outputs, domain_pred = self.model(inputs)
        outputs = softmax(outputs)
        return outputs

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
