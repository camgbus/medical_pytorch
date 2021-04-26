# Import needed libraries
import torch
import os
import numpy as np
import pandas as pd
import shutil
from mp.visualization.plot_results import plot_dataframe

def save_results(model, noise, paths, pathr, losses_train, losses_val, accuracy_train,
                 accuracy_det_train, accuracy_val, accuracy_det_val, losses_test, accuracy_test,
                 accuracy_det_test, losses_cum_train, losses_cum_val):
    r"""This function saves the results from a trained model, i.e. losses and accuracies."""
    print('Save trained model losses and accuracies..')
    torch.save(model.state_dict(), os.path.join(paths, 'model_state_dict.zip'))
    model_path = os.path.join(os.environ["OPERATOR_PERSISTENT_DIR"], noise)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    else:
        # Empty directory
        shutil.rmtree(model_path)
        os.makedirs(model_path)
    #torch.save(model, os.path.join(model_path, 'model.zip'))
    torch.save(model.state_dict(), os.path.join(model_path, 'model_state_dict.zip'))
    np.save(os.path.join(pathr, 'losses_train.npy'), np.array(losses_train))
    np.save(os.path.join(pathr, 'losses_cum_train.npy'), np.array(losses_cum_train))
    np.save(os.path.join(pathr, 'losses_validation.npy'), np.array(losses_val))
    np.save(os.path.join(pathr, 'losses_cum_validation.npy'), np.array(losses_cum_val))
    np.save(os.path.join(pathr, 'accuracy_train.npy'), np.array(accuracy_train))
    np.save(os.path.join(pathr, 'accuracy_detailed_train.npy'), np.array(accuracy_det_train))
    np.save(os.path.join(pathr, 'accuracy_validation.npy'), np.array(accuracy_val))
    np.save(os.path.join(pathr, 'accuracy_detailed_validation.npy'), np.array(accuracy_det_val))
    np.save(os.path.join(pathr, 'losses_test.npy'), np.array(losses_test))
    np.save(os.path.join(pathr, 'accuracy_test.npy'), np.array(accuracy_test))
    np.save(os.path.join(pathr, 'accuracy_detailed_test.npy'), np.array(accuracy_det_test))
    plot_dataframe(pd.DataFrame(losses_cum_train, columns = ['Epoch', 'Loss']),
        save_path = pathr, save_name = 'losses_train', title = 'Losses [train dataset]',
        x_name = 'Epoch', y_name = 'Loss', ending = '.png', ylog = False, figsize = (10,5),
        xints = float, yints = float)
    plot_dataframe(pd.DataFrame(accuracy_train, columns = ['Epoch', 'Accuracy']),
        save_path = pathr, save_name = 'accuracy_train', title = 'Accuracy [train dataset] in %',
        x_name = 'Epoch', y_name = 'Accuracy', ending = '.png', ylog = False, figsize = (10,5),
        xints = float, yints = float)
    plot_dataframe(pd.DataFrame(losses_cum_val, columns = ['Epoch', 'Loss']),
        save_path = pathr, save_name = 'losses_val', title = 'Losses [validation dataset]',
        x_name = 'Epoch', y_name = 'Loss', ending = '.png', ylog = False, figsize = (10,5),
        xints = float, yints = float)
    plot_dataframe(pd.DataFrame(accuracy_val, columns = ['Epoch', 'Accuracy']),
        save_path = pathr, save_name = 'accuracy_val', title = 'Accuracy [validation dataset] in %',
        x_name = 'Epoch', y_name = 'Accuracy', ending = '.png', ylog = False, figsize = (10,5),
        xints = float, yints = float)
    plot_dataframe(pd.DataFrame(losses_test, columns = ['Batch', 'Loss']),
        save_path = pathr, save_name = 'losses_test', title = 'Losses [test dataset]',
        x_name = 'Batch', y_name = 'Loss', ending = '.png', ylog = False, figsize = (10,5),
        xints = float, yints = float)
    plot_dataframe(pd.DataFrame(accuracy_test, columns = ['Batch', 'Accuracy']),
        save_path = pathr, save_name = 'accuracy_test', title = 'Accuracy [test dataset] in %',
        x_name = 'Batch', y_name = 'Accuracy', ending = '.png', ylog = False, figsize = (10,5),
        xints = int, yints = int)

def save_only_test_results(noise, path, losses_test, accuracy_test, accuracy_det_test):
    r"""This function saves the test results from a pretrained model, i.e. losses and accuracies."""
    print('Save pre-trained model losses and accuracies on test dataset..')
    np.save(os.path.join(path, 'losses_test.npy'), np.array(losses_test))
    np.save(os.path.join(path, 'accuracy_test.npy'), np.array(accuracy_test))
    np.save(os.path.join(path, 'accuracy_detailed_test.npy'), np.array(accuracy_det_test))
    plot_dataframe(pd.DataFrame(losses_test, columns = ['Batch', 'Loss']),
        save_path = path, save_name = 'losses_test', title = 'Losses [test dataset]',
        x_name = 'Batch', y_name = 'Loss', ending = '.png', ylog = False, figsize = (10,5),
        xints = float, yints = float)
    plot_dataframe(pd.DataFrame(accuracy_test, columns = ['Batch', 'Accuracy']),
        save_path = path, save_name = 'accuracy_test', title = 'Accuracy [test dataset] in %',
        x_name = 'Batch', y_name = 'Accuracy', ending = '.png', ylog = False, figsize = (10,5),
        xints = int, yints = int)