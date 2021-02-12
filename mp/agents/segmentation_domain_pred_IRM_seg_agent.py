import torch

from mp.agents.segmentation_domain_pred_agent import SegmentationDomainPredictionAgent
from mp.data.pytorch.domain_prediction_dataset_wrapper import DomainPredictionDatasetWrapper
from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax
from mp.eval.losses.losses_irm import IRMLossAbstract
from mp.utils.early_stopping import EarlyStopping
from mp.utils.helper_functions import zip_longest_with_cycle


# Dom Pred training schedule with limited IRM on segmentor only
class SegmentationDomainPredictionIRMAgent(SegmentationDomainPredictionAgent):
    r"""
    An Agent for segmentation models using a classifier for the domain space using the features from the encoder.
    Uses IRM on the segmentor but lambda is set to 1.0 during the whole training (IRM sub-stages are ignored).
    """

    def perform_stage1_training_epoch(self, optimizer,
                                      irm_loss_f_classifier,
                                      loss_f_domain_pred,
                                      train_dataloaders,
                                      alpha,
                                      print_run_loss=False):
        r"""Perform a stage 1 training epoch,
        meaning that the encoder, classifier and domain predictor are all trained together

        Args:
            irm_loss_f_classifier (IRMLossAbstract): the IRM loss function for the classifier
            irm_loss_f_domain_pred (IRMLossAbstract): the IRM loss function for the domain predictor
            print_run_loss (bool): whether a running loss should be tracked and printed.
        """
        acc = Accumulator('loss')
        # For each batch
        for data_list in zip_longest_with_cycle(*train_dataloaders):
            classifier_losses = []
            classifier_penalties = []
            domain_preds = []
            data_lengths = []  # Is used to produce the domain targets on the fly
            # For each dataloader
            for idx, data in enumerate(data_list):
                # Get data
                inputs, targets = self.get_inputs_targets(data)

                # Forward pass for the classification and domain prediction
                # Here we cannot use self.get_outputs(inputs)
                classifier_output, domain_pred = self.model(inputs)

                # Computing ERM and IRM terms for classification
                classifier_output = softmax(classifier_output)
                classifier_losses.append(irm_loss_f_classifier.erm(classifier_output, targets))
                classifier_penalties.append(irm_loss_f_classifier(classifier_output, targets))

                # Store predictions
                domain_preds.append(domain_pred)
                data_lengths.append(inputs.shape[0])

            # Domain prediction
            domain_preds = softmax(torch.cat(domain_preds, dim=0))
            domain_targets = self._create_domain_targets(data_lengths)

            # Optimization step
            optimizer.zero_grad()
            loss = irm_loss_f_classifier.finalize_loss(classifier_losses, classifier_penalties) + \
                   alpha * loss_f_domain_pred(domain_preds, domain_targets)

            loss.backward()
            optimizer.step()
            acc.add('loss', float(loss.detach().cpu()))

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

    def perform_stage2_training_epoch(self, optimizer_model,
                                      optimizer_domain_predictor,
                                      optimizer_encoder,
                                      irm_loss_f_classifier,
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
            classifier_penalties = []
            features = []
            data_lengths = []  # Is used to produce the domain targets on the fly
            # For each dataloader
            for data in data_list:
                # Get data
                inputs, targets = self.get_inputs_targets(data)

                # Forward pass for the classification
                # Here we cannot use self.get_outputs(inputs)
                feature = self.model.get_features_from_encoder(inputs)
                classifier_output = softmax(self.model.get_classification_from_features(feature))

                # Store losses and predictions
                # Computing ERM and IRM terms for classification
                classifier_output = softmax(classifier_output)
                classifier_losses.append(irm_loss_f_classifier.erm(classifier_output, targets))
                classifier_penalties.append(irm_loss_f_classifier(classifier_output, targets))
                features.append(feature)
                data_lengths.append(inputs.shape[0])

            # Model Optimization step
            optimizer_model.zero_grad()

            loss = irm_loss_f_classifier.finalize_loss(classifier_losses, classifier_penalties)
            acc.add('loss', float(loss.detach().cpu()))

            loss.backward(retain_graph=True)
            optimizer_model.step()

            # Domain Predictor Optimization step
            optimizer_domain_predictor.zero_grad()
            features = torch.cat(features, dim=0)
            domain_pred = self.model.get_domain_prediction_from_features(features.detach())

            domain_targets = self._create_domain_targets(data_lengths)
            domain_pred._requires_grad = True

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

    def train_with_early_stopping(self, results, optimizers, losses, train_dataloaders, train_dataset_names,
                                  early_stopping,
                                  init_epoch=0,
                                  run_loss_print_interval=10,
                                  eval_datasets=None, eval_interval=10,
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
        irm_loss_f_classifier, _, _ = losses
        irm_loss_f_classifier.penalty_weight = 1.

        return super().train_with_early_stopping(results, optimizers, losses, train_dataloaders, train_dataset_names,
                                                 early_stopping, init_epoch=init_epoch,
                                                 run_loss_print_interval=run_loss_print_interval,
                                                 eval_datasets=eval_datasets, eval_interval=eval_interval,
                                                 save_path=save_path, beta=beta)
