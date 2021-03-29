from sklearn.neural_network import MLPRegressor
import os 
import pickle
class Dice_predictor():
    '''a simple MLP Regressor model ,that predicts a dice score from a feature vector
    '''
    def __init__(self, features = [], add_to_name='', verbose=False):
        self.have_model = False
        self.regressor = None
        self.features = features
        
        path = os.path.join(os.environ['OPERATOR_PERSISTENT_DIR'],'dice_predictors')
        if not os.path.isdir(path):
            os.makedirs(path)

        self.path_to_model = os.path.join(path,
                    '{}.sav'.format(add_to_name)) 
        self.path_to_model_descr = os.path.join(path,
                    '{}_descr.txt'.format(add_to_name))
        self.verbose = verbose

    def predict(self, input):
        '''predicts a dice score, for given feature vector''' 
        if self.have_model:
            input = [input]
            output = self.regressor.predict(input)
            return output[0]
        else:
            print('Please load or train model first')
            raise RuntimeError

    def load(self):
        '''loads the model with given name'''
        try:
            self.regressor = pickle.load(open(self.path_to_model,'rb'))
        except:
            print('there is no model with this name, please it train first before loading')
            raise RuntimeError
        self.have_model = True 
        if self.verbose:
            self.print_description()

    def train(self,X_train,y_train,  data_descr='', model_descr='',
                **kwargs):
        if self.verbose:
            print('training model')

        self.regressor = MLPRegressor(**kwargs)
        self.regressor.fit(X_train,y_train)
        self.have_model = True

        regressor_score = self.regressor.score(X_train,y_train)
        if self.verbose:
            losses_string = 'The regressor has a score of {} in train data'.format(regressor_score)
            print(losses_string)

        with open(self.path_to_model,'wb') as saver:
            pickle.dump(self.regressor,saver)
        
        self._save_descr(data_descr,model_descr,regressor_score,**kwargs)

    def retrain(self,X_train,y_train,  data_descr='', model_descr='',
            **kwargs):
        if self.verbose:
            print('retraining model')

        self.regressor = MLPRegressor(**kwargs)
        self.regressor.fit(X_train,y_train)
        self.have_model = True

        regressor_score = self.regressor.score(X_train,y_train)
        if self.verbose:
            losses_string = 'The regressor has a score of {} in train data'.format(regressor_score)
            print(losses_string)

        with open(self.path_to_model,'wb') as saver:
            pickle.dump(self.regressor,saver)
    
        self._add_descr(data_descr,model_descr,regressor_score**kwargs)


    def _save_descr(self,data_d,model_d,score,**kwargs):
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
            file.write("regressor loss on train data : \n")
            file.write('{}'.format(score))
    
    def _add_descr(self,data_d,model_d,score,**kwargs):
        with open(self.path_to_model_descr,'a') as file:
            file.write(r'Model has been retrained, new settings: \n')
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
            file.write("regressor loss on train data : \n")
            file.write('{}'.format(score))

    def print_description(self):
        with open(self.path_to_model_descr,'r') as file:
            for line in file:
                print(line)