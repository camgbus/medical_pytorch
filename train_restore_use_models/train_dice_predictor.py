from mp.models.regression.dice_predictor import Dice_predictor
from mp.utils.feature_extractor import Feature_extractor
from mp.models.densities.density import Density_model

#which additional information shall be submitted for training
kwargs = {'solver'='adam','lr'='adaptive','hidden_layer_sizes'=(10,30,50,50,20)}
# a name to find model, and make it uniquely identifiable
ending_of_model = 'first_try'
#features to use for model
features=['density_distance','dice_scores','connected_components']

#which density to use for feature extraction and the model itself
model,add_to_name = 'gaussian_kernel',''
#input paths to data, that shall be used for training here:
list_of_paths = []
#input train labels for data, that shall be used for training, here:
y_train = None #depending on how is the training data stored ? 

#string of informations on data to store for density
data_describtion = ''
#string of informations on model to store for density 
model_describtion = ''

#initiate model
dice_pred = Dice_predictor(features=features,add_to_name=ending_of_model)

#get train data
density = Density_model(model=model,add_to_name=add_to_name)
feat_extr = Feature_extractor(density,features)
features = np.array(feat_extr.get_features_from_paths(list_of_paths))
X_train = np.array(features)
y_train = y_train #needs to get implemented or given from above

#train model
dice_pred.train(X_train,y_train, data_describtion, model_describtion,**kwargs)