import os
from itertools import zip_longest

import torch

from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax
from mp.eval.losses.loss_abstract import LossAbstract
from mp.utils.pytorch.pytorch_load_restore import save_optimizer_state, load_optimizer_state
from mp.utils.early_stopping import EarlyStopping


class SegmentationIRMGamesAgent(SegmentationAgent):
    r"""An Agent for segmentation models using IRM Games models."""

    def get_outputs(self, inputs, keep_grad_idx=None):
        r"""Applies a softmax transformation to the model outputs"""
        outputs = self.model.predict(inputs, keep_grad_idx=keep_grad_idx)
        outputs = softmax(outputs)
        return outputs

    def perform_training_epoch(self, optimizers, loss_f, train_dataloaders, print_run_loss=False):
        r"""Perform a training epoch

        Args:
            loss_f (LossAbstract): the loss
            print_run_loss (bool): whether a running loss should be tracked and
                printed.
        """
        acc = Accumulator('loss')
        for data_list in zip_longest(*train_dataloaders):
            losses = []

            for data_idx, (opt, data) in enumerate(zip(optimizers, data_list)):
                # Get data
                inputs, targets = self.get_inputs_targets(data)

                # Forward pass
                outputs = self.get_outputs(inputs, data_idx)

                opt.zero_grad()
                loss = loss_f(outputs, targets)
                # Optimization step
                loss.backward()
                opt.step()

                losses.append(loss)

            acc.add('loss', float(torch.stack(losses).detach().cpu().mean()))

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

    def train(self, results, optimizers, loss_f, train_dataloaders,
              init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
              eval_datasets=None, eval_interval=10,
              save_path=None, save_interval=10):
        r"""Train a model through its agent. Performs training epochs,
        tracks metrics and saves model states.

        Args:
            optimizers (list): a list of optimizers (one for each DL)
            loss_f (LossAbstract): the loss
            train_dataloaders (list): a list of Dataloader
        """

        # Model must be an IRMGamesModel and the nb of optimizers / Dataloaders must match the number of sub-models
        assert len(train_dataloaders) == len(self.model.models), "Nb of Dataloaders doe not match nb of sub-models"
        assert len(optimizers) == len(self.model.models), "Nb of optimizers doe not match nb of sub-models"

        if eval_datasets is None:
            eval_datasets = dict()

        if init_epoch == 0:
            self.track_metrics(init_epoch, results, loss_f, eval_datasets)
        for epoch in range(init_epoch, init_epoch + nr_epochs):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose

            self.perform_training_epoch(optimizers, loss_f, train_dataloaders, print_run_loss=print_run_loss)

            # Track statistics in results
            if (epoch + 1) % eval_interval == 0:
                self.track_metrics(epoch + 1, results, loss_f, eval_datasets)

            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1), optimizers)

    def train_with_early_stopping(self, results, optimizers, loss_f, train_dataloaders, early_stopping,
                                  init_epoch=0, run_loss_print_interval=10,
                                  eval_datasets=None, eval_interval=10,
                                  save_path=None, save_interval=10):
        r"""Train a model through its agent. Performs training epochs,
        tracks metrics and saves model states.

        Args:
            optimizers (list): a list of optimizers (one for each DL)
            loss_f (LossAbstract): the loss
            train_dataloaders (list): a list of Dataloader
            early_stopping (EarlyStopping): the early stopping criterion
        """

        # Model must be an IRMGamesModel and the nb of optimizers / Dataloaders must match the number of sub-models
        assert len(train_dataloaders) == len(self.model.models), "Nb of Dataloaders doe not match nb of sub-models"
        assert len(optimizers) == len(self.model.models), "Nb of optimizers doe not match nb of sub-models"

        if eval_datasets is None:
            eval_datasets = dict()

        early_stopping.reset()

        epoch = init_epoch
        if epoch == 0:
            self.track_metrics(init_epoch, results, loss_f, eval_datasets)
        for epoch in range(epoch, 1 << 32):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose

            self.perform_training_epoch(optimizers, loss_f, train_dataloaders, print_run_loss=print_run_loss)

            # Track statistics in results
            if (epoch + 1) % eval_interval == 0:
                self.track_metrics(epoch + 1, results, loss_f, eval_datasets)

                # Stop stage if needed
                if not early_stopping.check_results(results, epoch + 1):
                    break

        # Save agent and optimizer state
        if save_path is not None:
            self.save_state(save_path, 'epoch_{}'.format(epoch + 1), optimizers)

        return epoch + 1,

    def save_state(self, states_path, state_name, optimizers=None, overwrite=False):
        r"""Saves an agent state. Raises an error if the directory exists and
        overwrite=False.
        """
        # This is here because we need to save multiple optimizers

        if states_path is not None:
            # We take care of saving the optimizers ourselves
            super().save_state(states_path, state_name)
            if optimizers is not None:
                state_full_path = os.path.join(states_path, state_name)
                for idx, optimizer in enumerate(optimizers):
                    save_optimizer_state(optimizer, f'optimizer{idx}', state_full_path)

    def restore_state(self, states_path, state_name, optimizers=None):
        r"""Tries to restore a previous agent state, consisting of a model
        state and the content of agent_state_dict. Returns whether the restore
        operation  was successful.
        """
        # This is here because we need to restore multiple optimizers

        try:
            # We take care of restoring the optimizers ourselves
            if not super().restore_state(states_path, state_name):
                return False

            if self.verbose:
                print('Trying to restore optimizer states...'.format(state_name))

            if optimizers is not None:
                state_full_path = os.path.join(states_path, state_name)
                for idx, optimizer in enumerate(optimizers):
                    load_optimizer_state(optimizer, f'optimizer{idx}', state_full_path, device=self.device)
            if self.verbose:
                print('Optimizer states {} were restored'.format(state_name))
            return True
        except:
            print('Complete state {} could not be restored'.format(state_name))
            return False
