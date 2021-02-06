from mp.models.densities.density import Density_model
from mp.utils.intensities import get_intensities

#which kind of model is to be used
model = 'gaussian_kernel'
#which additional information shall be submitted for training
kwargs = {'bw':20}
# a name to find model, and make it uniquely identifiable
ending_of_model = ''

#iput paths to data, that shall be used for density estimation here:
list_of_paths = []

#string of informations on data to store for density
data_describtion = ''
#string of informations on model to store for density 
model_describtion = ''

density_model = Density_model(model=model,add_to_name=ending_of_model)
intensity_values = get_intensities(list_of_paths)
density_model.train_density(intensity_values,
                    data_describtion,model_describtion,**kwargs)