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
        self.path_to_model = os.path.join('storage','models','dice_predictor',
                    '{}_{}.sav'.format(features,add_to_name)) 
        self.path_to_model_descr = os.path.join('storage','models','dice_predictor',
                    '{}_{}_descr.txt'.format(features,add_to_name))
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
                retrain=False, **kwargs):
        if os.path.isfile(self.path_to_model) and not retrain:
            print('model already exists and retrain is set to false')
            raise RuntimeError
        else: 
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
            
            self._save_descr(data_descr,model_descr,regressor_score**kwargs)


    def _save_descr(self,data_d,model_d,score,**kwargs):
        with open(self.path_to_model_descr,'w') as file:
            file.write(r'Data describtion: \n')
            file.write(data_d)
            file.write(r'Model describtion: \n')
            file.write(model_d)
            file.wirte(r'train parameter : \n')
            file.write('{}'.format(kwargs))
            file.wirte(r'regressor loss on train data : \n')
            file.write('{}'.format(score))


    def print_description(self):
        with open(self.path_to_model_descr,'w') as file:
            for line in file:
                print(line)
