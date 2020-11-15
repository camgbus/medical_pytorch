# ------------------------------
# A standard regression agent.
# ------------------------------

from mp.agents.agent import Agent
from mp.eval.inference.predict import softmax
import torch

class RegressionAgent(Agent):
    r"""An Agent for regression models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, optimizer, loss_f, train_dataloader,
              nr_epochs=100, save_path=None, save_interval=10):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """
        losses = list()
        losses_cum = list()
        for epoch in range(nr_epochs):
            msg = "Running epoch "
            msg += str(epoch + 1) + " of " + str(nr_epochs) + "."
            print (msg, end = "\r")
            epoch_loss = list()
            total = 0
            correct = 0
            for idx, (x, y) in enumerate(train_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = loss_f(yhat, y)
                total += y.size(0)
                epoch_loss.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(epoch_loss)
            losses_cum.append([epoch+1, sum(epoch_loss) / total])
            print('Epoch --> Loss: {} --> {:.4}.'.format(epoch + 1,
                                                         sum(epoch_loss) / total))
            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                print('Saving current state after epoch: {}'.format(epoch + 1))
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                optimizer, overwrite=True)

        # Return losses
        return losses, losses_cum

    def test(self, loss_f, test_dataloader):
        losses = list()
        accuracy = list()
        total = 0
        losses_cum = 0
        correct = 0
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = loss_f(yhat, y)
                losses.append([idx+1, loss])
                total += y.size(0)
                losses_cum += loss
                # Round prediction to 1 decimal float numbers
                rounded_yhat = torch.round(yhat * 10**1) / (10**1)
                correct += (rounded_yhat == y).sum().item()
                accuracy.append([idx+1, 100 * (rounded_yhat == y).sum().item() / y.size(0)])
        print('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
        print('Accuracy of the regression model on the test set: %d %%' % (
            100 * correct / total))

        # Return losses
        return losses, accuracy

        