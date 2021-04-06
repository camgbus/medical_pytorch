from sklearn.neighbors import KernelDensity
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
    
    def __init__(self, clusters=False, model='gaussian_kernel',add_to_name='standart',verbose=False):
        self.model = model
        self.density = None
        self.clusters = clusters
        self.verbose = verbose

        density_path = os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'density_models')
        if not os.path.isdir(density_path):
            os.makedirs(density_path)
        
        if self.clusters:
            self.path_to_model = os.path.join(density_path,
                    self.model+'_cluster_'+add_to_name+'.sav')
            self.path_to_model_descr = os.path.join(density_path,
                    self.model+'_cluster_'+add_to_name+'_descr.txt')
            self.path_to_int_values = os.path.join(density_path,
                    self.model+'_cluster_'+add_to_name+'int_values.npy')
        else:
            self.path_to_model = os.path.join(density_path,
                    self.model+'_no_cluster_'+add_to_name+'.sav')
            self.path_to_model_descr = os.path.join(density_path,
                    self.model+'_no_cluster_'+add_to_name+'_descr.txt')
            self.path_to_int_values = os.path.join(density_path,
                    self.model+'_no_cluster_'+add_to_name+'int_values.npy')

    def load_density(self):
        '''loads density, depending on clusters,model,add_to_name'''
        if self.model == 'gaussian_kernel':
            try:
                self.density = pickle.load(open(self.path_to_model,'rb'))
            except:
                print('there is no density model with this name, please it train first before loading')
                raise RuntimeError
        if self.verbose:
            self.print_description()
        
    
    def train_density(self, int_values, data_descr='', 
                        model_descr='',**kwargs):
        '''trains a density for the given int_values, saves a descr with same name as model 
        as .txt file
        
        Args:
            int_values (ndarray(numbers)): a one-dim array of intensity values
            data_descr (str): a string, that describes the data used
            model_descr (str): a string, that describes the model used
            retrain (bool): in order to not accidently overwrite model retrain has to be set to true 
                manually if we want to retrain
            **kwargs : arguments used for model training, depend on external implementation of model used
        '''
        if self.model == 'gaussian_kernel':
            if self.clusters : 
                raise NotImplementedError
            else :
                data = np.reshape(int_values, newshape=(-1,1))
                self.density = KernelDensity(kernel='gaussian', **kwargs).fit(data)

        pickle.dump(self.density,open(self.path_to_model,'wb'))
        self._save_descr(data_descr,model_descr,**kwargs)

    def retrain_density(self, int_values, add_data_descr='', 
                        add_model_descr='',**kwargs):
        '''trains a density for the given int_values, saves a descr with same name as model 
        as .txt file
        
        Args:
            int_values (ndarray(numbers)): a one-dim array of intensity values
            data_descr (str): a string, that describes the new data used
            model_descr (str): a string, that describes the model used
            retrain (bool): in order to not accidently overwrite model retrain has to be set to true 
                manually if we want to retrain
            **kwargs : arguments used for model training, depend on external implementation of model used
        '''
        if self.model == 'gaussian_kernel':
            if self.clusters : 
                raise NotImplementedError
            else :
                data = np.reshape(int_values, newshape=(-1,1))
                self.density = KernelDensity(kernel='gaussian', **kwargs).fit(data)
        
        pickle.dump(self.density,open(self.path_to_model,'wb'))
        self._update_descr(add_data_descr,add_model_descr,**kwargs)

    def get_values(self,steps=0.001):
        '''gets values of the density in the interval [0,1] in order to compute 
        distance to other density
        Args:
            steps (float): the inverse of how many values shall be taken
        
        returns (ndarray): a array of the computed density values in the intervall'''
        points = np.reshape(np.arange(start=0,stop=1,step=steps),(-1,1))
        if self.model == 'gaussian_kernel':
            density_values = np.exp(self.density.score_samples(points))
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

    def _update_descr(self,add_data_descr,add_model_descr,**kwargs):
        with open(self.path_to_model_descr,'a') as file:
            file.write("New Settings for Data and model: \n")
            file.write("Data describtion: \n")
            file.write(add_data_descr)
            file.write("\n")
            file.write("Model describtion: \n")
            file.write(add_model_descr)
            file.write("\n")
            file.write("Model Settings: \n")
            file.write('{}'.format(kwargs))
    
    def print_description(self):
        with open(self.path_to_model_descr,'w') as file:
            for line in file:
                print(line)
    
    def plot_density(self,steps=0.001):
        x = np.arange(start=0,stop=1,step=steps)
        y = self.get_values()
        plt.plot(x,y)
        plt.show()



    





