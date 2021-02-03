# ------------------------------------------------------------------------------
# A standard segmentation agent, which performs softmax in the outputs.
# ------------------------------------------------------------------------------

from mp.agents.agent import Agent
from mp.eval.inference.predict import softmax
import torch
import os
from torchvision import transforms
import numpy as np
from mp.paths import telegram_login
from mp.utils.update_bots.telegram_bot import TelegramBot
from mp.utils.agents.save_restore import save_state as external_save_state
from mp.utils.agents.save_restore import restore_state as external_restore_state
import sys 

class SegmentationAgent(Agent):
    r"""An Agent for segmentation models."""
    def __init__(self, *args, **kwargs):
        if 'metrics' not in kwargs:
            kwargs['metrics'] = ['ScoreDice', 'ScoreIoU']
        super().__init__(*args, **kwargs)

    def get_outputs(self, inputs):
        r"""Applies a softmax transformation to the model outputs"""
        outputs = self.model(inputs)
        outputs = softmax(outputs)
        return outputs

class UNet2DAgent(SegmentationAgent):
    r"""An Agent for UNet2D models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
              val_dataloader, nr_epochs=100, start_epoch=0, save_path=None,
              losses=list(), losses_cum=list(), losses_val=list(), losses_cum_val=list(),
              save_interval=10, msg_bot=True, bot_msg_interval=10):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """
        assert start_epoch < nr_epochs, 'Start epoch needs to be smaller than the number of epochs!'
        if msg_bot == True:
            self.bot.send_msg('Start training the model for {} epochs..'.format(nr_epochs-start_epoch))
            
        for epoch in range(start_epoch, nr_epochs):
            msg = "Running epoch "
            msg += str(epoch + 1) + " of " + str(nr_epochs) + "."
            print (msg, end = "\r")
            epoch_loss = list()
            #total = number of slices used
            total = 0
            for idx, (x, y) in enumerate(train_dataloader):
                # trying to get right data format for bce computation
                # x, y = x.type(torch.float32), y.type(torch.float32)
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.get_outputs(x)
                sys.exit("he told me to do this")
                # i assume the dataloader loads the images in normal format 
                # not as multichannel in which case #torch.max(y, 1)[1]                
                loss = loss_f(yhat, y.float()) 
                total += y.size(0)
                epoch_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(epoch_loss)
            losses_cum.append([epoch+1, sum(epoch_loss) / total])

            # Validate current model based on validation dataloader
            epoch_loss_val = list()
            total_val = 0
            with torch.no_grad():
                for idx, (x, y) in enumerate(val_dataloader):
                    x_val, y_val = x.to(self.device), y.to(self.device)
                    yhat_val = self.get_outputs(x_val)
                    loss = loss_f(yhat_val,y_val.float())
                    total_val += y_val.size(0)
                    epoch_loss_val.append(loss.item())
                losses_val.append(epoch_loss_val)
                losses_cum_val.append([epoch+1, sum(epoch_loss_val) / total_val])

            print('Epoch --> Loss --> : {} --> {:.4} .\n'
                   'Val_Loss --> :{:.4} )'.format(epoch + 1,
                                                    sum(epoch_loss) / total,
                                                    sum(epoch_loss_val) / total_val))
            # Save agent and optimizer state
            if (epoch + 1) % save_interval == 0 and save_path is not None:
                print('Saving current state after epoch: {}.'.format(epoch + 1))
                self.save_state(save_path, 'epoch_{}'.format(epoch + 1),
                                optimizer, True, losses, losses_cum, losses_val,
                                losses_cum_val)
                
        # Return losses
        return losses, losses_cum, losses_val, losses_cum_val

    def test(self, loss_f, test_dataloader, msg_bot=True):
        if msg_bot == True:
            self.bot.send_msg('Start testing the model..')
        losses = list()
        total = 0
        losses_cum = 0
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.get_outputs(x)
                loss = loss_f(yhat, y.float())
                losses.append([idx+1, loss.item()])
                total += y.size(0)
                losses_cum += loss.item()
        print('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
        if msg_bot == True:
            self.bot.send_msg('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
            
        # Return losses
        return losses, losses_cum
