# ------------------------------
# A standard regression agent.
# ------------------------------

from mp.agents.agent import Agent
import torch
import numpy as np
from mp.paths import telegram_login
from mp.utils.update_bots.telegram_bot import TelegramBot

class RegressionAgent(Agent):
    r"""An Agent for regression models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot = TelegramBot(telegram_login)

    def train(self, optimizer, loss_f, train_dataloader,
              val_dataloader, nr_epochs=100, save_path=None,
              save_interval=10, msg_bot=True, bot_msg_interval=10):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """
        if msg_bot == True:
            self.bot.send_msg('Start training the model for {} epochs..'.format(nr_epochs))
        losses = list()
        losses_cum = list()
        losses_val = list()
        losses_cum_val = list()
        accuracy = list()
        accuracy_detailed = list()
        accuracy_val = list()
        accuracy_val_detailed = list()
        for epoch in range(nr_epochs):
            msg = "Running epoch "
            msg += str(epoch + 1) + " of " + str(nr_epochs) + "."
            print (msg, end = "\r")
            epoch_loss = list()
            results_y = list()
            results_yhat = list()
            results_mod_yhat = list()
            total = 0
            correct = 0
            for idx, (x, y) in enumerate(train_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = loss_f(yhat, y)
                total += y.size(0)
                epoch_loss.append(loss.item())
                mod_yhat = self.group_output(yhat)
                correct += (mod_yhat == y).sum().item()
                results_y.extend(y.cpu().detach().numpy().tolist())
                results_yhat.extend(yhat.cpu().detach().numpy().tolist())
                results_mod_yhat.extend(mod_yhat.cpu().detach().numpy().tolist())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(epoch_loss)
            losses_cum.append([epoch+1, sum(epoch_loss) / total])
            accuracy.append([epoch+1, 100 * correct / total])
            accuracy_detailed.append(list(zip(results_y, results_yhat, results_mod_yhat)))

            # Validate current model based on validation dataloader
            epoch_loss_val = list()
            results_y_val = list()
            results_yhat_val = list()
            results_mod_yhat_val = list()
            total_val = 0
            correct_val = 0
            with torch.no_grad():
                for idx, (x, y) in enumerate(val_dataloader):
                    x_val, y_val = x.to(self.device), y.to(self.device)
                    yhat_val = self.model(x_val)
                    loss = loss_f(yhat_val, y_val)
                    total_val += y_val.size(0)
                    epoch_loss_val.append(loss.item())
                    mod_yhat_val = self.group_output(yhat_val)
                    correct_val += (mod_yhat_val == y_val).sum().item()
                    results_y_val.extend(y_val.cpu().detach().numpy().tolist())
                    results_yhat_val.extend(yhat_val.cpu().detach().numpy().tolist())
                    results_mod_yhat_val.extend(mod_yhat_val.cpu().detach().numpy().tolist())
                losses_val.append(epoch_loss_val)
                losses_cum_val.append([epoch+1, sum(epoch_loss_val) / total_val])
                accuracy_val.append([epoch+1, 100 * correct_val / total_val])
                accuracy_val_detailed.append(list(zip(results_y_val, results_yhat_val,
                results_mod_yhat_val)))
        
            print(('Epoch --> Loss --> Accuracy: {} --> {:.4} --> {:.4}%.\n'
                   'Val_Loss --> Val_Accuracy: {:.4} --> {:.4}%.').format(epoch + 1,
                                                    sum(epoch_loss) / total,
                                                    100 * correct / total,
                                                    sum(epoch_loss_val) / total_val,
                                                    100 * correct_val / total_val))
            if (epoch + 1) % bot_msg_interval == 0 and msg_bot:
                self.bot.send_msg(('Epoch --> Loss --> Accuracy: {} --> {:.4} --> {:.4}%.\n'
                                   'Val_Loss --> Val_Accuracy: {:.4} --> {:.4}%.').format(epoch + 1,
                                                                    sum(epoch_loss) / total,
                                                                    100 * correct / total,
                                                                    sum(epoch_loss_val) / total_val,
                                                                    100 * correct_val / total_val))
            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                print('Saving current state after epoch: {}.'.format(epoch + 1))
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                optimizer, overwrite=True)

        # Return losses
        return losses, losses_cum, losses_cum_val, accuracy, accuracy_detailed, accuracy_val, accuracy_val_detailed

    def test(self, loss_f, test_dataloader, msg_bot=True):
        if msg_bot == True:
            self.bot.send_msg('Start testing the model..')
        losses = list()
        accuracy = list()
        accuracy_detailed = list()
        total = 0
        losses_cum = 0
        correct = 0
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = loss_f(yhat, y)
                losses.append([idx+1, loss.item()])
                total += y.size(0)
                losses_cum += loss.item()
                mod_yhat = self.group_output(yhat)
                correct += (mod_yhat == y).sum().item()
                accuracy.append([idx+1, 100 * (mod_yhat == y).sum().item() / y.size(0)])
                accuracy_detailed.extend(list(zip(y.cpu().numpy().tolist(),
                                                  yhat.cpu().numpy().tolist(),
                                                  mod_yhat.cpu().numpy().tolist())))
        print('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
        print('Accuracy of the regression model on the test set: %d %%' % (
            100 * correct / total))
        if msg_bot == True:
            self.bot.send_msg('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
            self.bot.send_msg('Accuracy of the regression model on the test set: %d %%' % (
            100 * correct / total))
            
        # Return losses
        return losses, accuracy, accuracy_detailed

    def group_output(self, yhat):
        r"""This function returns the corresponding group based on
            yhat (given defined ranges)."""
        yhat = yhat.detach().cpu().numpy()
        yhat_mod = list()
        for y in yhat:
            if y <= 0.3:
                yhat_mod.append([0.2])
                continue
            if y > 0.3 and y <= 0.5:
                yhat_mod.append([0.4])
                continue
            if y > 0.5 and y <= 0.7:
                yhat_mod.append([0.6])
                continue
            if y > 0.7 and y <= 0.9:
                yhat_mod.append([0.8])
                continue
            if y > 0.9:
                yhat_mod.append([1.0])
                continue
        return torch.tensor(yhat_mod).to(self.device)