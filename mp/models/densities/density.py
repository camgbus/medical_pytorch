from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
import pickle      
import numpy as np  
import os 
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

class Density_model():
    '''the class that is responsible for computation of densities 
        Densities will be for the intervall [0,1]

        model: str; in order to tell code, which model is being used
        clusters (not implemented) if multiple densities are computed, one for each cluster
        add_to_name = str ; give model a name to find it again and not overwrite models
        verbose = bool ; if model shall be verbose '''
    
    def __init__(self, model_name='gaussian_mixture',add_to_name='standart',verbose=False):
        self.model_name = model_name
        self.model = None
        self.verbose = verbose
        self.density_values = None

        density_path = os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'density_models')

        if not os.path.isdir(density_path):
            os.makedirs(density_path)

        self.path_to_model = os.path.join(density_path,
                self.model_name+'_no_cluster_'+add_to_name+'.sav')
        self.path_to_model_descr = os.path.join(density_path,
                self.model_name+'_no_cluster_'+add_to_name+'_descr.txt')
        self.path_to_dens_values = os.path.join(density_path,
                self.model_name+'_no_cluster_'+add_to_name+'_values.npy')

    def load_density(self):
        '''loads density'''
        self.model = pickle.load(open(self.path_to_model,'rb'))
        if self.verbose:
            self.print_description()
        
    def load_density_values(self):
        '''loads the values of the density, which is all that is needed for 
        coputations and less memory heavy'''
        self.density_values = np.load(self.path_to_dens_values,allow_pickle=True)

    def train_density(self, int_values, data_descr='', 
                        model_descr='',**kwargs):
        '''trains a density for the given int_values, saves a descr with same name as model 
        as .txt file, also saves the model and the values of the density over the intervall 
        [0,1]
        Args:
            int_values (ndarray(numbers)): a one-dim array of intensity values
            data_descr (str): a string, that describes the data used
            model_descr (str): a string, that describes the model used
            retrain (bool): in order to not accidently overwrite model retrain has to be set to true 
                manually if we want to retrain
            **kwargs : arguments used for model training, depend on external implementation of model used
        '''
        if self.model_name == 'gaussian_kernel':
            data = np.reshape(int_values, newshape=(-1,1))
            if kwargs:
                self.model = KernelDensity(kernel='gaussian', **kwargs).fit(data)
            else:
                self.model = KernelDensity(kernel='gaussian', bandwidth=0.005).fit(data)
        if self.model_name == 'gaussian_mixture':
            data = np.reshape(int_values, newshape=(-1,1))
            if kwargs:
                self.model = GaussianMixture(**kwargs).fit(data)
            else:
                self.model = GaussianMixture(n_components=2).fit(data)
            
        self._save_density_values()
        pickle.dump(self.model,open(self.path_to_model,'wb'))
        self._save_descr(data_descr,model_descr,**kwargs)

    def get_values(self,steps=0.001):
        '''gets values of the density in the interval [0,1] in order to compute 
        distance to other density
        Args:
            steps (float): the inverse of how many values shall be taken
        
        returns (ndarray): a array of the computed density values in the intervall'''
        points = np.reshape(np.arange(start=0,stop=1,step=steps),(-1,1))
        density_values = np.exp(self.model.score_samples(points))
        return density_values

    def _save_descr(self,data_d,model_d,**kwargs):
        with open(self.path_to_model_descr,'w') as file:
            file.write("Data describtion: \n")
            file.write(data_d)
            file.write("\n")
            file.write("Model describtion: \n")
            file.write(model_d)
            file.write("\n")
            file.write("Model Settings: \n")
            file.write('{}'.format(kwargs))
    
    def _save_density_values(self):
        '''computes the values of the density over the intervall [0,1] and 
        saves them in a numpy file'''
        density_values = self.get_values()
        np.save(self.path_to_dens_values,density_values)

    def print_description(self):
        with open(self.path_to_model_descr,'w') as file:
            for line in file:
                print(line)
    
    def plot_density(self,steps=0.001):
        x = np.arange(start=0,stop=1,step=steps)
        y = self.get_values()
        plt.plot(x,y)
        plt.show()



    





