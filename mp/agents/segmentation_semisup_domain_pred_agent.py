import torch

from mp.agents.segmentation_domain_pred_agent import SegmentationDomainPredictionAgent
from mp.data.pytorch.domain_prediction_dataset_wrapper import DomainPredictionDatasetWrapper
from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax
from mp.utils.early_stopping import EarlyStopping
from mp.utils.helper_functions import zip_longest_with_cycle


class SegmentationSemisupDomainPredictionAgent(SegmentationDomainPredictionAgent):
    r"""
    An Agent for segmentation models using a classifier for the domain space using the features from the encoder
    with an IRM loss for the segmentor.
    """

    def perform_stage2_training_epoch(self, optimizer_model,
                                      optimizer_domain_predictor,
                                      optimizer_encoder,
                                      loss_f_classifier,
                                      loss_f_domain_pred,
                                      loss_f_encoder,
                                      train_dataloaders_seg,
                                      train_dataloaders_dp,
                                      beta,
                                      print_run_loss=False):
        r"""Perform a stage 2 training epoch,
        meaning that the encoder, classifier and domain predictor are all trained one after the other

        Args:
            print_run_loss (bool): whether a running loss should be tracked and printed.
        """
        # The main difference in this semi-sup version is that
        # the domain predictor is trained using another set of dataloaders

        acc = Accumulator('loss')
        # We zip the dataloaders for segmentor and domain predictor
        # Each of these lists of dataloaders contains a dataloader per dataset
        for data_list_seg, data_list_dp in zip(zip_longest_with_cycle(*train_dataloaders_seg),
                                               zip_longest_with_cycle(*train_dataloaders_dp)):
            classifier_losses = []

            # For each dataloader
            for data_seg in data_list_seg:
                # Get data for the segmentor
                inputs, targets = self.get_inputs_targets(data_seg)

                # Forward pass for the classification
                # Here we cannot use self.get_outputs(inputs)
                feature = self.model.get_features_from_encoder(inputs)
                outputs = softmax(self.model.get_classification_from_features(feature))

                # Store losses and predictions
                classifier_losses.append(loss_f_classifier(outputs, targets))

            # Model Optimization step
            optimizer_model.zero_grad()

            loss = torch.stack(classifier_losses, dim=0).mean()
            acc.add('loss', float(loss.detach().cpu()))

            loss.backward(retain_graph=True)
            optimizer_model.step()

            # Domain Predictor Optimization step
            data_lengths = []  # Is used to produce the domain targets on the fly
            features = []
            # For each dataloader
            for data_dp in data_list_dp:
                # Get data
                inputs, _ = self.get_inputs_targets(data_dp)
                features.append(self.model.get_features_from_encoder(inputs))
                data_lengths.append(inputs.shape[0])

            optimizer_domain_predictor.zero_grad()
            features = torch.cat(features, dim=0)
            domain_pred = self.model.get_domain_prediction_from_features(features.detach())

            domain_targets = self._create_domain_targets(data_lengths)

            loss_dm = loss_f_domain_pred(domain_pred, domain_targets)
            loss_dm.backward(retain_graph=False)
            optimizer_domain_predictor.step()

            # Encoder Optimization step based on domain prediction loss
            features = []
            for data_dp in data_list_dp:
                # Get data
                inputs, _ = self.get_inputs_targets(data_dp)
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

    def train(self, results,
              optimizers,
              losses,
              train_dataloaders_seg,
              train_dataloaders_dp,
              train_dataset_names,
              init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
              eval_datasets=None, eval_interval=10,
              save_path=None, save_interval=10,
              alpha=50.0, beta=1., stage1_epochs=100):
        # TODO maybe implement this later
        raise NotImplementedError

    def train_with_early_stopping(self, results, optimizers, losses,
                                  train_dataloaders_seg, train_dataloaders_dp, train_dataset_names,
                                  early_stopping,
                                  init_epoch=0,
                                  run_loss_print_interval=10,
                                  eval_datasets_seg=None, eval_datasets_dp=None, eval_interval=10,
                                  save_path=None,
                                  alpha=1.0, beta=10.):
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
                self.track_metrics(epoch + 1, results, loss_f_classifier, eval_datasets_seg)
                self.track_domain_prediction_accuracy(epoch + 1, results, train_datasets_wrappers)
                return early_stopping_criterion.check_results(results, epoch + 1)

            return True

        stage1_optimizer, optimizer_model, optimizer_domain_predictor, optimizer_encoder = optimizers

        if eval_datasets_seg is None:
            eval_datasets_seg = dict()
        if eval_datasets_dp is None:
            eval_datasets_dp = dict()

        # Creating the wrappers for the datasets
        train_datasets_wrappers = {
            key: DomainPredictionDatasetWrapper(eval_datasets_dp[key], train_dataset_names.index(key[0]))
            for key in eval_datasets_dp if key[0] in train_dataset_names}
        # # Also add the train and val splits from the segmentor
        # train_datasets_wrappers_stage1 = {
        #     key: DomainPredictionDatasetWrapper(eval_datasets_seg[key], train_dataset_names.index(key[0]))
        #     for key in eval_datasets_seg if key[0] in train_dataset_names and "test" not in key[1]}

        loss_f_classifier, loss_f_domain_pred, loss_f_encoder = losses
        early_stopping.reset_counter()

        # Tracking metrics at step 0
        epoch = stage1_last_epoch = stage2_last_epoch = stage3_last_epoch = init_epoch
        self.track_metrics(epoch, results, loss_f_classifier, eval_datasets_seg)
        self.track_domain_prediction_accuracy(epoch, results, train_datasets_wrappers)

        # Stage 1
        for epoch in range(epoch, 1 << 32):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0 and self.verbose

            self.perform_stage1_training_epoch(stage1_optimizer,
                                               loss_f_classifier,
                                               loss_f_domain_pred,
                                               train_dataloaders_seg,
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
                                               train_dataloaders_seg,
                                               train_dataloaders_dp,
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
                                              list({key[0] + "_val_dp" for key in train_datasets_wrappers}),
                                              metric_min_delta=early_stopping.metric_min_delta)
        for epoch in range(epoch + 1, 1 << 32):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0 and self.verbose

            self.perform_stage3_training_epoch(optimizer_domain_predictor,
                                               loss_f_domain_pred,
                                               train_dataloaders_dp,
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
