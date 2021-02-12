import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings. simplefilter(action='ignore', category=Warning)
import pandas as pd 
import numpy as np 
from tensorflow.keras.models import model_from_json
import pickle
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class Prescription(object):
    

    def __init__(self, model_json_file, model_weights_file, scaler_file, enc_input_file, enc_output_file):

        with open(model_json_file, 'r') as f:
            json_model = f.read()
            self.loaded_model = model_from_json(json_model)

        with open(scaler_file,'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(enc_input_file,'rb') as f:
            self.enc_input = pickle.load(f)
        
        with open(enc_output_file,'rb') as f:
            self.enc_output = pickle.load(f)
        

        self.loaded_model.load_weights(model_weights_file)


    def predict_prescription(self, age, temperature, speciality, findings):

        self.all_preds = []

        for finding in findings:

            cont = [[age, temperature]]
            cont = self.scaler.transform(cont)

            cat = [[speciality, finding]]
            cat = self.enc_input.transform(cat)

            inp = np.concatenate([cont, cat], axis = 1)
            inp = inp.reshape(1,-1)

            pred = self.loaded_model.predict(inp)
            ind = np.argsort(pred)[0][-2:]
            preds = []
            for i in ind:
                out = np.zeros((1,78))
                out[0][i] = 1
                preds.append(self.enc_output.inverse_transform(out)[0][0])

            self.all_preds.append(preds)
        
        self.all_preds = np.unique(np.ravel(self.all_preds))

        return self.all_preds

