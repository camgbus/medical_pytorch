import torch

from mp.agents.segmentation_semisup_domain_pred_agent import SegmentationSemisupDomainPredictionAgent
from mp.data.pytorch.domain_prediction_dataset_wrapper import DomainPredictionDatasetWrapper
from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax
from mp.utils.domain_prediction_utils import perform_stage1_training_epoch
from mp.utils.early_stopping import EarlyStopping
from mp.utils.helper_functions import zip_longest_with_cycle


class SegmentationSemisupDomainPredictionIRMAgent(SegmentationSemisupDomainPredictionAgent):
    r"""An Agent for segmentation models using a classifier for the domain space using the features from the encoder"""

    def perform_stage2_training_epoch(self, optimizer_model,
                                      optimizer_domain_predictor,
                                      optimizer_encoder,
                                      irm_loss_f_classifier,
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
            classifier_penalties = []

            # For each dataloader
            for data_seg in data_list_seg:
                # Get data for the segmentor
                inputs, targets = self.get_inputs_targets(data_seg)

                # Forward pass for the classification
                # Here we cannot use self.get_outputs(inputs)
                feature = self.model.get_features_from_encoder(inputs)
                classifier_output = softmax(self.model.get_classification_from_features(feature))

                # Store losses and predictions
                classifier_losses.append(irm_loss_f_classifier.erm(classifier_output, targets))
                classifier_penalties.append(irm_loss_f_classifier(classifier_output, targets))

            # Model Optimization step
            optimizer_model.zero_grad()

            loss = irm_loss_f_classifier.finalize_loss(classifier_losses, classifier_penalties)
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
