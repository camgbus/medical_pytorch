import torch
from mp.agents.segmentation_agent import SegmentationAgent
from mp.eval.accumulator import Accumulator


class SegmentationIRMAgent(SegmentationAgent):
    r"""An Agent for segmentation models using IRM."""

    def perform_training_epoch(self, optimizer, loss_f, train_dataloaders,
                               print_run_loss=False, penalty_weight=1.):
        r"""Perform a training epoch

        Args:
            print_run_loss (bool): whether a runing loss should be tracked and
                printed.
        """
        acc = Accumulator('loss')
        # FIXME for now assume that there are as many batches in each dataloader
        for data_list in zip(*train_dataloaders):
            losses = []
            penalties = []
            for data in data_list:
                # Get data
                inputs, targets = self.get_inputs_targets(data)

                # Forward pass
                outputs = self.get_outputs(inputs)

                losses.append(loss_f(outputs, targets))
                penalties.append(self._penalty(loss_f, outputs, targets))

            # Optimization step
            optimizer.zero_grad()
            loss = torch.stack(losses, dim=0).mean() + penalty_weight * torch.stack(penalties, dim=0).mean()
            if penalty_weight > 1.:
                loss /= penalty_weight

            loss.backward()
            optimizer.step()
            acc.add('loss', float(loss.detach().cpu()))

        if print_run_loss:
            print('\nRunning loss: {}'.format(acc.mean('loss')))

    def train(self, results, optimizer, loss_f, train_dataloaders,
              init_epoch=0, nr_epochs=100, run_loss_print_interval=10,
              eval_datasets=None, eval_interval=10,
              save_path=None, save_interval=10,
              penalty_weight=1e5, penalty_anneal_iters=1000):
        r"""Train a model through its agent. Performs training epochs,
        tracks metrics and saves model states.
        """
        if eval_datasets is None:
            eval_datasets = dict()

        if init_epoch == 0:
            self.track_metrics(init_epoch, results, loss_f, eval_datasets)
        for epoch in range(init_epoch, init_epoch + nr_epochs):
            print_run_loss = (epoch + 1) % run_loss_print_interval == 0
            print_run_loss = print_run_loss and self.verbose
            self.perform_training_epoch(optimizer, loss_f, train_dataloaders,
                                        print_run_loss=print_run_loss,
                                        penalty_weight=penalty_weight if epoch >= penalty_anneal_iters else 1.)

            # Track statistics in results
            if (epoch + 1) % eval_interval == 0:
                self.track_metrics(epoch + 1, results, loss_f, eval_datasets)

            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1), optimizer)

    @staticmethod
    def _penalty(loss_f, output, target):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = loss_f(output * scale, target)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)
