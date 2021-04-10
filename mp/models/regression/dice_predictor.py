import os 
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

class Dice_predictor():
    '''a simple linear Ridge Regressor model ,that predicts a dice score from a feature vector
    '''
    def __init__(self, features = [], version='', verbose=False):

        self.regressor = None
        self.features = features
        
        path = os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'dice_predictors')
        if not os.path.isdir(path):
            os.makedirs(path)

        self.scaler = None

        self.path_to_scaler = os.path.join(path,'{}_scaler.sav')
        self.path_to_model = os.path.join(path,
                    '{}.sav'.format(version)) 
        self.path_to_model_descr = os.path.join(path,
                    '{}_descr.txt'.format(version))
        self.verbose = verbose


    def load(self):
        '''loads the model with given name'''
        try:
            self.regressor = pickle.load(open(self.path_to_model,'rb'))
            self.scaler = pickle.load(open(self.path_to_scaler,'rb'))
        except:
            print('there is no model with this name, please it train first before loading')
            raise RuntimeError
        if self.verbose:
            self.print_description()

    def train(self,X_train,y_train,  data_descr='', model_descr='',
                **kwargs):
        if self.verbose:
            print('training model')

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        if kwargs:
            self.regressor = Ridge(**kwargs)
        else:
            self.regressor = Ridge(normalize=False)
        
        self.regressor.fit(X_train_scaled,y_train)

        regressor_score = self.regressor.score(X_train_scaled,y_train)
        regressor_l2_loss = self.l2_loss(X_train_scaled,y_train)

        losses_string = 'The regressor has a score of {} in train data and an l2 loss of{}'.format(regressor_score,regressor_l2_loss)
        if self.verbose:
            print(losses_string)

        with open(self.path_to_model,'wb') as saver:
            pickle.dump(self.regressor,saver)
        
        with open(self.path_to_scaler,'wb') as saver: 
            pickle.dump(self.scaler,saver)

        self._save_descr(data_descr,model_descr,losses_string,**kwargs)

    def predict(self, input):
        '''predicts dice scores, for given feature matrix''' 
        input = self.scaler.transform(input)
        output = self.regressor.predict(input)
        return output[0]

    def l2_loss(self, X_scaled, truth):
        n = len(truth)
        y_pred = self.regressor.predict(X_scaled)
        loss_sum = np.sum((y_pred - truth)**2)
        return loss_sum/n

    def _save_descr(self,data_d,model_d,losses_string,**kwargs):
        with open(self.path_to_model_descr,'w') as file:
            file.write("Data describtion: \n")
            file.write(data_d)
            file.write("\n")
            file.write("Used features: \n")
            file.write('{}'.format(self.features))
            file.write("\n")
            file.write("Model describtion: \n")
            file.write(model_d)
            file.write("\n")
            file.write("train parameter : \n")
            file.write('{}'.format(kwargs))
            file.write("\n")
            file.write(losses_string)

    def print_description(self):
        with open(self.path_to_model_descr,'r') as file:
            for line in file:
                print(line)