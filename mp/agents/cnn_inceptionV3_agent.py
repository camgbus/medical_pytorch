# ------------------------------------------------------------------
# A model agent for CNN InceptionV3 model.
# ------------------------------------------------------------------

from mp.agents.agent import Agent
from mp.eval.inference.predict import softmax
import torch
from torchvision import transforms
from sklearn.metrics import r2_score

class CNNInceptionV3Agent(Agent):
    r"""An Agent for a CNN InceptionV3 model."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, optimizer, loss_f, train_dataloader,
              nr_epochs=100, save_path=None, save_interval=10):
        r"""Train a model through its agent. Performs training epochs, 
        tracks metrics and saves model states.
        """
        losses = list()
        losses_cum = list()
        # Necessary for input image into InceptionV3 model
        preprocess = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for epoch in range(nr_epochs):
            msg = "Running epoch "
            msg += str(epoch + 1) + " of " + str(nr_epochs) + "."
            print (msg, end = "\r")
            epoch_loss = list()
            total = 0
            correct = 0
            for idx, (x, y) in enumerate(train_dataloader):
                y = y.to(self.device)
                for i in range(list(x.size())[0]):
                    x[i] = preprocess(x[i])
                x = x.unsqueeze(0) # create a mini-batch as expected by the model
                x = x.to(self.device)
                yhat = self.model(x)
                loss = loss_f(yhat, y)
                total += y.size(0)
                epoch_loss.append(loss.item())
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
        total = 0
        losses_cum = 0
        # Necessary for input image into InceptionV3 model
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with torch.no_grad():
            for idx, (x, y) in enumerate(test_dataloader):
                y = y.to(self.device)
                x = preprocess(x)
                x = x.unsqueeze(0) # create a mini-batch as expected by the model
                x = x.to(self.device)
                yhat = self.model(x)
                loss = loss_f(yhat, y)
                losses.append([idx+1, loss.item()])
                total += y.size(0)
                losses_cum += loss.item()
        print('Testset --> Overall Loss: {:.4}.'.format(losses_cum / total))

        # Return losses
        return losses