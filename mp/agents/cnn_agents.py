# ------------------------------
# A AlexNet agent.
# ------------------------------

from mp.agents.agent import Agent
import torch
from torchvision import transforms
import numpy as np
from mp.paths import telegram_login
from mp.utils.update_bots.telegram_bot import TelegramBot

class AlexNetAgent(Agent):
    r"""An Agent for AlexNet models."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bot = TelegramBot(telegram_login)

    def preprocess(self, img_tensor):
        r"""Transforms an image based on the desired input of
        AlexNet.
        """
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        batch_size = list(img_tensor.size())[0]
        new_img = preprocess(img_tensor[0].cpu().detach()).unsqueeze(0)
        for batch in range(1, batch_size):
            img = img_tensor[batch]
            img = preprocess(img.cpu().detach()).unsqueeze(0) # preprocess image and add batch dimension
            new_img = torch.cat((new_img, img), 0)
        return new_img.to(self.device)

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
            total = 0
            correct = 0
            for idx, (x, y) in enumerate(train_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                x = self.replicating_image(x, 3)
                x = self.preprocess(x)
                yhat = self.model(x)
                loss = loss_f(yhat, torch.max(y, 1)[1])
                total += y.size(0)
                epoch_loss.append(loss.item())
                _, mod_yhat = torch.max(yhat, 1)
                _, labels = torch.max(y, 1)
                correct += (mod_yhat == labels).sum().item()
                results_y.extend(y.cpu().detach().numpy().tolist())
                results_yhat.extend(yhat.cpu().detach().numpy().tolist())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(epoch_loss)
            losses_cum.append([epoch+1, sum(epoch_loss) / total])
            accuracy.append([epoch+1, 100 * correct / total])
            accuracy_detailed.append(list(zip(results_y, results_yhat)))

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
                    x_val = self.replicating_image(x_val, 3)
                    x_val = self.preprocess(x_val)
                    yhat_val = self.model(x_val)
                    loss = loss_f(yhat_val, torch.max(y_val, 1)[1])
                    total_val += y_val.size(0)
                    epoch_loss_val.append(loss.item())
                    _, mod_yhat_val = torch.max(yhat_val, 1)
                    _, labels_val = torch.max(y_val, 1)
                    correct_val += (mod_yhat_val == labels_val).sum().item()
                    results_y_val.extend(y_val.cpu().detach().numpy().tolist())
                    results_yhat_val.extend(yhat_val.cpu().detach().numpy().tolist())
                losses_val.append(epoch_loss_val)
                losses_cum_val.append([epoch+1, sum(epoch_loss_val) / total_val])
                accuracy_val.append([epoch+1, 100 * correct_val / total_val])
                accuracy_val_detailed.append(list(zip(results_y_val, results_yhat_val)))

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
        return losses, losses_cum, accuracy, accuracy_val, accuracy_detailed

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
                x = self.replicating_image(x, 3)
                x = self.preprocess(x)
                yhat = self.model(x)
                loss = loss_f(yhat, torch.max(y, 1)[1])
                losses.append([idx+1, loss.item()])
                total += y.size(0)
                losses_cum += loss.item()
                _, mod_yhat = torch.max(yhat, 1)
                _, labels = torch.max(y, 1)
                correct += (mod_yhat == labels).sum().item()
                accuracy.append([idx+1, 100 * (mod_yhat == labels).sum().item() / y.size(0)])
                accuracy_detailed.extend(list(zip(y.cpu().numpy().tolist(),
                                                  yhat.cpu().numpy().tolist())))
        print('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
        print('Accuracy of the regression model on the test set: %d %%' % (
            100 * correct / total))
        if msg_bot == True:
            self.bot.send_msg('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
            self.bot.send_msg('Accuracy of the regression model on the test set: %d %%' % (
            100 * correct / total))
            
        # Return losses
        return losses, accuracy, accuracy_detailed

class NetAgent(Agent):
    r"""An Agent for AlexNet models."""
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
            total = 0
            correct = 0
            for idx, (x, y) in enumerate(train_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)
                loss = loss_f(yhat, torch.max(y, 1)[1])
                total += y.size(0)
                epoch_loss.append(loss.item())
                _, mod_yhat = torch.max(yhat, 1)
                _, labels = torch.max(y, 1)
                correct += (mod_yhat == labels).sum().item()
                results_y.extend(y.cpu().detach().numpy().tolist())
                results_yhat.extend(yhat.cpu().detach().numpy().tolist())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(epoch_loss)
            losses_cum.append([epoch+1, sum(epoch_loss) / total])
            accuracy.append([epoch+1, 100 * correct / total])
            accuracy_detailed.append(list(zip(results_y, results_yhat)))

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
                    loss = loss_f(yhat_val, torch.max(y_val, 1)[1])
                    total_val += y_val.size(0)
                    epoch_loss_val.append(loss.item())
                    _, mod_yhat_val = torch.max(yhat_val, 1)
                    _, labels_val = torch.max(y_val, 1)
                    correct_val += (mod_yhat_val == labels_val).sum().item()
                    results_y_val.extend(y_val.cpu().detach().numpy().tolist())
                    results_yhat_val.extend(yhat_val.cpu().detach().numpy().tolist())
                losses_val.append(epoch_loss_val)
                losses_cum_val.append([epoch+1, sum(epoch_loss_val) / total_val])
                accuracy_val.append([epoch+1, 100 * correct_val / total_val])
                accuracy_val_detailed.append(list(zip(results_y_val, results_yhat_val)))

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
                loss = loss_f(yhat, torch.max(y, 1)[1])
                losses.append([idx+1, loss.item()])
                total += y.size(0)
                losses_cum += loss.item()
                _, mod_yhat = torch.max(yhat, 1)
                _, labels = torch.max(y, 1)
                correct += (mod_yhat == labels).sum().item()
                accuracy.append([idx+1, 100 * (mod_yhat == labels).sum().item() / y.size(0)])
                accuracy_detailed.extend(list(zip(y.cpu().numpy().tolist(),
                                                  yhat.cpu().numpy().tolist())))
        print('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
        print('Accuracy of the regression model on the test set: %d %%' % (
            100 * correct / total))
        if msg_bot == True:
            self.bot.send_msg('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))
            self.bot.send_msg('Accuracy of the regression model on the test set: %d %%' % (
            100 * correct / total))
            
        # Return losses
        return losses, accuracy, accuracy_detailed

def replicating_image(self, img, nr):
    replicat_img = img.repeat(1, nr, 1, 1)
    return replicat_img