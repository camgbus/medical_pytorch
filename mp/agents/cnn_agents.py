# --------------------------------------
# A cnn agents for classification task.
# --------------------------------------

from mp.agents.agent import Agent
import torch
from mp.paths import telegram_login
from mp.utils.update_bots.telegram_bot import TelegramBot
from mp.utils.agents.save_restore import save_state as external_save_state
from mp.utils.agents.save_restore import restore_state as external_restore_state

class NetAgent(Agent):
    r"""An Agent for CNN models."""
    def __init__(self, lr_decay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot = TelegramBot(telegram_login)
        self.lr_decay = lr_decay

    def save_state(self, states_path, state_name, optimizer=None, overwrite=False,
                   losses_train=None, losses_cum_train=None, losses_val=None, 
                   losses_cum_val=None, accuracy_train=None, accuracy_det_train=None,
                   accuracy_val=None, accuracy_det_val=None):
        r"""Saves an agent state. Raises an error if the directory exists and 
        overwrite=False. Saves all further results like losses and accuracies as
        .npy files.
        """
        external_save_state(self, states_path, state_name, optimizer, overwrite,
                            losses_train, losses_cum_train, losses_val, losses_cum_val,
                            accuracy_train, accuracy_det_train, accuracy_val, accuracy_det_val)

    def restore_state(self, states_path, state_name, optimizer=None):
        r"""Tries to restore a previous agent state, consisting of a model 
        state and the content of agent_state_dict. Returns whether the restore 
        operation  was successful. Further the results will be loaded as well,
        i.e. losses and accuracies.
        """
        return external_restore_state(self, states_path, state_name, optimizer)

    def train(self, optimizer, loss_f, train_dataloader,
              val_dataloader, decay_rate=0.1, decay_type='exp_decay', nr_epochs=100, start_epoch=0, save_path=None,
              losses=list(), losses_cum=list(), losses_val=list(), losses_cum_val=list(),
              accuracy=list(), accuracy_detailed=list(), accuracy_val=list(),
              accuracy_val_detailed=list(), save_interval=10,
              msg_bot=True, bot_msg_interval=10, store_data=False):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """
        assert start_epoch < nr_epochs, 'Start epoch needs to be smaller than the number of epochs!'
        if msg_bot == True:
            self.bot.send_msg('Start training the model for {} epochs..'.format(nr_epochs-start_epoch))

        # Define exponential lr decay optimizer
        if self.lr_decay:
            scheduler = {'exp_decay'  : torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate),
                         'step_decay' : torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=decay_rate),
                         'mstep_decay': torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,100], gamma=decay_rate),
                         'plat_decay' : torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')}
            
        for epoch in range(start_epoch, nr_epochs):
            msg = "Running epoch "
            msg += str(epoch + 1) + " of " + str(nr_epochs) + "."
            print (msg, end = "\r")
            epoch_loss = list()
            results_y = list()
            results_yhat = list()
            total = 0
            correct = 0
            for (x, y) in train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = loss_f(yhat, torch.max(y, 1)[1])
                total += y.size(0)
                epoch_loss.append(loss.item())
                _, mod_yhat = torch.max(yhat, 1)
                _, labels = torch.max(y, 1)
                correct += (mod_yhat == labels).sum().item()
                if store_data:
                    results_y.extend(y.cpu().detach().numpy().tolist())
                    results_yhat.extend(yhat.cpu().detach().numpy().tolist())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses_cum.append([epoch+1, sum(epoch_loss) / total])
            accuracy.append([epoch+1, 100 * correct / total])
            if store_data:
                losses.append(epoch_loss)
                accuracy_detailed.append(list(zip(results_y, results_yhat)))

            # Validate current model based on validation dataloader
            epoch_loss_val = list()
            results_y_val = list()
            results_yhat_val = list()
            total_val = 0
            correct_val = 0
            with torch.no_grad():
                for (x, y) in val_dataloader:
                    x_val, y_val = x.to(self.device), y.to(self.device)
                    yhat_val = self.model(x_val)
                    loss = loss_f(yhat_val, torch.max(y_val, 1)[1])
                    total_val += y_val.size(0)
                    epoch_loss_val.append(loss.item())
                    _, mod_yhat_val = torch.max(yhat_val, 1)
                    _, labels_val = torch.max(y_val, 1)
                    correct_val += (mod_yhat_val == labels_val).sum().item()
                    if store_data:
                        results_y_val.extend(y_val.cpu().detach().numpy().tolist())
                        results_yhat_val.extend(yhat_val.cpu().detach().numpy().tolist())
                losses_cum_val.append([epoch+1, sum(epoch_loss_val) / total_val])
                accuracy_val.append([epoch+1, 100 * correct_val / total_val])
                if store_data:
                    losses_val.append(epoch_loss_val)
                    accuracy_val_detailed.append(list(zip(results_y_val, results_yhat_val)))

            # Perform lr decay
            if self.lr_decay:
                if decay_type == 'plat_decay':
                    scheduler[decay_type].step(sum(epoch_loss_val))
                else:
                    scheduler[decay_type].step()

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
                                optimizer, True, losses, losses_cum, losses_val,
                                losses_cum_val, accuracy, accuracy_detailed, accuracy_val,
                                accuracy_val_detailed)

        # Return losses
        return losses, losses_cum, losses_val, losses_cum_val, accuracy, accuracy_detailed, accuracy_val, accuracy_val_detailed

    def test(self, loss_f, test_dataloader, msg_bot=True, store_data=False):
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
                loss = loss_f(yhat, torch.max(y, 1)[1])
                losses.append([idx+1, loss.item()])
                total += y.size(0)
                losses_cum += loss.item()
                _, mod_yhat = torch.max(yhat, 1)
                _, labels = torch.max(y, 1)
                correct += (mod_yhat == labels).sum().item()
                accuracy.append([idx+1, 100 * (mod_yhat == labels).sum().item() / y.size(0)])
                if store_data:
                    accuracy_detailed.extend(list(zip(y.cpu().numpy().tolist(),
                                                    yhat.cpu().numpy().tolist())))
        print('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
        print('Accuracy of the cnn model on the test set: %d %%' % (
            100 * correct / total))
        if msg_bot == True:
            self.bot.send_msg('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
            self.bot.send_msg('Accuracy of the cnn model on the test set: %d %%' % (
            100 * correct / total))
            
        # Return losses
        return losses, losses_cum, accuracy, accuracy_detailed