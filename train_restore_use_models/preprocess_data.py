# Import needed libraries
from mp.data.datasets.dataset_JIP_cnn import JIPDataset

def preprocess_data(config):
    r"""This function is used to load the original data from the workflow and preprocesses it
        by saving it in the preprocessed workflow."""
    
    JIP = JIPDataset(img_size=config['input_shape'], gpu=True, cuda=config['device'], msg_bot = config['msg_bot'])
    return JIP.preprocess()