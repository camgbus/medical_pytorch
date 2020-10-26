import torch

from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator
from mp.eval.inference.predict import softmax
from mp.eval.losses.loss_abstract import LossAbstract


# FIXME refactor agents for IRM and IRM Games
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
        # FIXME for now assume that there are as many batches in each dataloader
        for data_list in zip(*train_dataloaders):
            losses = []

            for opt in optimizers:
                opt.zero_grad()

            for data_idx, data in enumerate(data_list):
                # Get data
                inputs, targets = self.get_inputs_targets(data)

                # Forward pass
                outputs = self.get_outputs(inputs, data_idx)

                losses.append(loss_f(outputs, targets))

            # Optimization step
            for opt, loss in zip(optimizers, losses):
                loss.backward()
                opt.step()

            acc.add('loss', float(torch.stack(losses).detach().cpu().mean()))

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

    def train(self, results, optimizers, loss_f, train_dataloaders,
              init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
              eval_datasets=None, eval_interval=10,
              save_path=None, save_interval=10,
              penalty_weight=1e5, penalty_anneal_iters=1000):
        r"""Train a model through its agent. Performs training epochs,
        tracks metrics and saves model states.

        Args:
            optimizers (list): a list of optimizers (one for each DL)
            loss_f (LossAbstract): the loss
            train_dataloaders (list): a list of Dataloaders
        """
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
                # FIXME add the possibility to save a list of optimizers
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1))
