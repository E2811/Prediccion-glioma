import pandas as pd
import numpy as np
from models import svm_model
import pickle 

class SVM_glioma():

    def __init__(self,path_model,path_encoder,continous_vars=['Age_at_diagnosis'], categorical_variables=['Race'], training_bool=False):
        ''' Clase para implementar el SVM 
        params
        --------
        path_model: ruta donde se encuentra el modelo alojado 
        continous_vars: lista con las columnas de tipo continuo en el data set
        categorical_variables: lista con las columnas de tipo categorico en el data set
        training_bool: Boleano verdadero si queremos reentrenar el modelo y guardar la variable '''
        self.path_model = path_model
        self.path_encoder = path_encoder
        self.training_bool = training_bool
        self.continous_vars = continous_vars
        self.categorical_variables = categorical_variables
        self.trained_model = None

    def read_model(self):
        ''' Función para leer el modelo ya entrenado '''
        self.trained_model = pickle.load(open(self.path_model, 'rb'))
        return self.trained_model

    def training(self,X_train, y_train, X_test, y_test):
        ''' Función para entrenar el modelo nuevamente con otros datos o parametros '''
        model_svm, metrics, roc_auc, fpr, tpr = svm_model(X_train, y_train, X_test, y_test,self.continous_vars, self.categorical_variables)
        self.trained_model = model_svm
        return model_svm
            
    def predict(self,X_val):
        ''' Función para predecir sobre datos introducidos '''
        encoder = pickle.load(open(self.path_encoder, 'rb'))
        X_val_transform = encoder.transform(X_val)
        self.read_model()
        y_pred = self.trained_model.predict(X_val_transform)
        return y_pred

    def save_model(self):
        ''' Función para guardar el nuevo modelo '''
        pickle.dump(self.trained_model, open(self.path_model, 'wb'))

    def main(self,X_val, X_train=None, y_train=None, X_test=None, y_test=None):
        if self.training_bool:
            self.training(X_train, y_train, X_test, y_test)
            self.save_model()
        y_pred = self.predict(X_val)
        return y_pred
