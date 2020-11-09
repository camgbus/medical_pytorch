from itertools import zip_longest

from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator
from mp.eval.losses.losses_irm import IRMLossAbstract
from mp.utils.helper_functions import zip_longest_with_cycle


class SegmentationIRMAgent(SegmentationAgent):
    r"""An Agent for segmentation models using IRM."""

    def perform_training_epoch(self, optimizer, irm_loss_f, train_dataloaders, print_run_loss=False):
        r"""Perform a training epoch

        Args:
            irm_loss_f (IRMLossAbstract): the IRM loss
            print_run_loss (bool): whether a running loss should be tracked and
                printed.
        """
        acc = Accumulator('loss')
        for data_list in zip_longest_with_cycle(*train_dataloaders):
            losses = []
            penalties = []
            for data in data_list:
                if data is None:
                    # The Dataloader corresponding to this data is has less examples than the others
                    continue

                # Get data
                inputs, targets = self.get_inputs_targets(data)

                # Forward pass
                outputs = self.get_outputs(inputs)

                losses.append(irm_loss_f.erm(outputs, targets))
                penalties.append(irm_loss_f(outputs, targets))

            # Optimization step
            optimizer.zero_grad()
            loss = irm_loss_f.finalize_loss(losses, penalties)

            loss.backward()
            optimizer.step()
            acc.add('loss', float(loss.detach().cpu()))

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

    def train(self, results, optimizer, irm_loss_f, train_dataloaders,
              init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
              eval_datasets=None, eval_interval=10,
              save_path=None, save_interval=10,
              penalty_weight=1e5, penalty_anneal_iters=1000):
        r"""Train a model through its agent. Performs training epochs,
        tracks metrics and saves model states.

        Args:
            irm_loss_f (IRMLossAbstract): the IRM loss
            train_dataloaders (list): a list of Dataloaders
        """
        if eval_datasets is None:
            eval_datasets = dict()

        if init_epoch == 0:
            # This penalty weight attribute needs to get reset between runs
            irm_loss_f.penalty_weight = 1.
            self.track_metrics(init_epoch, results, irm_loss_f, eval_datasets)
        for epoch in range(init_epoch, init_epoch + nr_epochs):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose

            # Update the penalty weight for the IRM loss if necessary
            if epoch == penalty_anneal_iters:
                irm_loss_f.penalty_weight = penalty_weight

            self.perform_training_epoch(optimizer, irm_loss_f, train_dataloaders, print_run_loss=print_run_loss)

            # Track statistics in results
            if (epoch + 1) % eval_interval == 0:
                self.track_metrics(epoch + 1, results, irm_loss_f, eval_datasets)

            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1), optimizer)
