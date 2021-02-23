from mp.models.densities.density import Density_model
from mp.utils.intensities import get_intensities
import os

#which kind of model is to be used
model = 'gaussian_kernel'
#which additional information shall be submitted for training
kwargs = {'bw':20}
# a name to find model, and make it uniquely identifiable
ending_of_model = 'first_try'

#input paths to data, that shall be used for density estimation here:
list_of_paths = [os.path.join('storage','data','UKF2')]

#string of informations on data to store for density
data_describtion = 'simply the UKF2 data, not resized, but already unpacked '
#string of informations on model to store for density 
model_describtion = 'gaussian kernel with bw 20'

#initialise density model
density_model = Density_model(model=model,add_to_name=ending_of_model)
#get the intensity values from the images 
intensity_values = get_intensities(list_of_paths)
#train density model
density_model.train_density(intensity_values,
                    data_describtion,model_describtion,**kwargs)

density_model.plot_density()
