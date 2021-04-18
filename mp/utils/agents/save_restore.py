import torch
import os
import shutil
import numpy as np
from mp.utils.load_restore import pkl_dump, pkl_load
from mp.utils.pytorch.pytorch_load_restore import save_model_state, load_model_state, save_optimizer_state, load_optimizer_state

def save_state(self, states_path, state_name, optimizer=None, overwrite=False,
               losses_train=None, losses_cum_train=None, losses_val=None, 
               losses_cum_val=None, accuracy_train=None, accuracy_det_train=None,
               accuracy_val=None, accuracy_det_val=None):
    r"""Saves an agent state. Raises an error if the directory exists and 
    overwrite=False. Saves all further results like losses and accuracies as
    .npy files.
    """
    if states_path is not None:
        state_full_path = os.path.join(states_path, state_name)
        if os.path.exists(state_full_path):
            if not overwrite:
                raise FileExistsError
            shutil.rmtree(state_full_path)
        os.makedirs(state_full_path)
        save_model_state(self.model, 'model_state_dict', state_full_path)
        pkl_dump(self.agent_state_dict, 'agent_state_dict', state_full_path)
        if optimizer is not None:
            save_optimizer_state(optimizer, 'optimizer', state_full_path)
        if losses_train is not None:
            np.save(os.path.join(state_full_path, 'losses_train.npy'), np.array(losses_train))
        if losses_cum_train is not None:
            np.save(os.path.join(state_full_path, 'losses_cum_train.npy'), np.array(losses_cum_train))
        if losses_val is not None:
            np.save(os.path.join(state_full_path, 'losses_validation.npy'), np.array(losses_val))
        if losses_cum_val is not None:
            np.save(os.path.join(state_full_path, 'losses_cum_validation.npy'), np.array(losses_cum_val))
        if accuracy_train is not None:
            np.save(os.path.join(state_full_path, 'accuracy_train.npy'), np.array(accuracy_train))
        if accuracy_det_train is not None:
            np.save(os.path.join(state_full_path, 'accuracy_detailed_train.npy'), np.array(accuracy_det_train))
        if accuracy_val is not None:
            np.save(os.path.join(state_full_path, 'accuracy_validation.npy'), np.array(accuracy_val))
        if accuracy_det_val is not None:
            np.save(os.path.join(state_full_path, 'accuracy_detailed_validation.npy'), np.array(accuracy_det_val))

def restore_state(self, states_path, state_name, optimizer=None):
    r"""Tries to restore a previous agent state, consisting of a model 
    state and the content of agent_state_dict. Returns whether the restore 
    operation  was successful. Further the results will be loaded as well,
    i.e. losses and accuracies.
    """
    state_full_path = os.path.join(states_path, state_name)
    try:
        correct_load = load_model_state(self.model, 'model_state_dict', state_full_path, device=self.device)
        assert correct_load
        agent_state_dict = pkl_load('agent_state_dict', state_full_path)
        assert agent_state_dict is not None
        self.agent_state_dict = agent_state_dict
        if optimizer is not None:
            load_optimizer_state(optimizer, 'optimizer', state_full_path, device=self.device)
        if self.verbose:
            print('State {} was restored'.format(state_name))
        print("Loading model state dict was successful.")
        losses_train = np.load(os.path.join(state_full_path, 'losses_train.npy'), allow_pickle=True)
        losses_cum_train = np.load(os.path.join(state_full_path, 'losses_cum_train.npy'), allow_pickle=True)
        losses_val = np.load(os.path.join(state_full_path, 'losses_validation.npy'), allow_pickle=True)
        losses_cum_val = np.load(os.path.join(state_full_path, 'losses_cum_validation.npy'), allow_pickle=True)
        accuracy_train = np.load(os.path.join(state_full_path, 'accuracy_train.npy'), allow_pickle=True)
        accuracy_det_train = np.load(os.path.join(state_full_path, 'accuracy_detailed_train.npy'), allow_pickle=True)
        accuracy_val = np.load(os.path.join(state_full_path, 'accuracy_validation.npy'), allow_pickle=True)
        accuracy_det_val = np.load(os.path.join(state_full_path, 'accuracy_detailed_validation.npy'), allow_pickle=True)
        print("Loading losses and accuracies was succesful.")
        return True, (losses_train, losses_cum_train, losses_val, losses_cum_val, accuracy_train, accuracy_det_train, accuracy_val, accuracy_det_val)
    except:
        print('State {} could not be restored'.format(state_name))
        return False, None