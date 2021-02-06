from sklearn.neighbors import KernelDensity
import pickle      
import numpy as np  
import os 
from skimage.measure import label, regionprops

class Density_model():
    '''the class that is responsible for computation of densities 
        Densities will be for the intervall [0,1]

        model: str; in order to tell code, which model is being used
        clusters (not implemented) if multiple densities are computed, one for each cluster
        add_to_name = str ; give model a name to find it again and not overwrite models
        verbose = bool ; if model shall be verbose '''
    
    def __init__(self, clusters=False, model='gaussian_kernel',add_to_name='',verbose=False):
        self.model = model
        self.have_model = False
        self.density = None
        self.clusters = clusters
        self.verbose = verbose

        if self.clusters:
            self.path_to_model = os.path.join('storage','models','density_models',
                    self.model+'_cluster_'+add_to_name+'.sav')
            self.path_to_model_descr = os.path.join('storage','models','density_models',
                    self.model+'_cluster_'+add_to_name+'_descr.txt')
        else:
            self.path_to_model = os.path.join('storage','models','density_models',
                    self.model+'_no_cluster_'+add_to_name+'.sav')
            self.path_to_model_descr = os.path.join('storage','models','density_models',
                    self.model+'_no_cluster_'+add_to_name+'_descr.txt')


    def load_density(self):
        '''loads density, depending on clusters,model,add_to_name'''
        if self.model == 'gaussian_kernel':
            try:
                self.density = pickle.load(open(self.path_to_model,'rb'))
                self.have_model = True
            except:
                print('there is no density model with this name, please it train first before loading')
                raise RuntimeError
        if self.verbose:
            self.print_description()
        
    
    def train_density(self, int_values, data_descr='', 
                        model_descr='',retrain=False,**kwargs):
        '''trains a density for the given int_values, saves a descr with same name as model 
        as .txt file'''
        #if model already exists and retrain s not true, overwriting is stopped
        if os.path.isfile(self.path_to_model) and not retrain:
            print('Model already exists, and retrain is set to false')
            raise RuntimeError
        if self.model == 'gaussian_kernel':
            if self.clusters : 
                raise NotImplementedError
            else :
                data = np.reshape(int_values, newshape=(-1,1))
                self.density = KernelDensity(kernel='gaussian', **kwargs).fit(data)

        self.have_model = True
        pickle.dump(self.density,open(self.path_to_model))
        self._save_descr(data_descr,model_descr)

    def get_values(self,steps=0.001):
        '''gets values of the density in the interval [0,1] in order to compute 
        distance to other density
        steps = float; the inverse of how many values shall to taken
        
        returns : np.array; a array of the computed density values in the intervall'''
        points = np.reshape(np.arange(start=0,stop=1,step=steps),(-1,1))
        if self.model == 'gaussian_kernel':
            density_values = np.exp(self.density.score_samples(points))
        return density_values

    def _save_descr(self,data_d,model_d):
        with open(self.path_to_model_descr,'w') as file:
            file.write(r'Data describtion: \n')
            file.write(data_d)
            file.write(r'Model describtion: \n')
            file.write(model_d)

    def print_description(self):
        with open(self.path_to_model_descr,'w') as file:
            for line in file:
                print(line)

    





