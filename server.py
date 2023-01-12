# -*- coding: utf-8 -*-
"""
Created on Mon May 30 14:38:48 2022

@author: user
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import math
from pathlib import Path
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import glob
import tensorflow as tf
from ThermalClient import ThermalClient
from Client import Client
from TargetClient import TargetClient

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class server():
    def __init__(self):
        conf = self.get_configuration()
        self.clients = conf['clients']
        self.num_of_clients = conf['number_of_clients']
        self.server_dir = conf['server_dir']
        self.Initiate = True
        
        self.common_features_nodes = conf['common_features_nodes']
        self.current_server_epoch = 1
        self.TrainingModel = conf['TrainingModel']
        self.server_epochs = conf['server_epochs']
        
        # Uncomment only this block 
        # f = open('monitor_server.txt', 'w+')
        # f.write('0')
        # f = open('monitor_clients.txt', 'w+')
        # f = open('monitor_clients.txt', 'w')
        # f.close()
        
        
        # Path("").mkdir(parents=True, exist_ok=True)
        # Path("clients.csv").mkdir(parents=True, exist_ok=True)
    
    def get_configuration(self):
        with open('config.json') as json_file:
             conf = json.load(json_file)
        return conf
    
    def Equalize_weights(self,standard_weights,local_weights):
        
        for i in range(0,len(standard_weights)):
            if (local_weights[i].shape != standard_weights[i].shape):
                print(local_weights[i].shape ,' Local ------------------')
                print(standard_weights[i].shape ,' Standard ------------------')
                temp = local_weights[i][0]
              
                for ii in range(1, standard_weights[i].shape[0]):
                     temp = np.append(temp, local_weights[i][ii], axis = 0)
                
                temp = temp.reshape(standard_weights[i].shape)
                local_weights[i] =  temp #local_weights[:standard_weights[i].shape[0]]
        #         for ii in range(len(server_weights[i]), len(local_weights[i])):
        #              temp = np.append(temp, [local_weights[i][ii]], axis = 0)
        #         server_weights[i] = temp
        return local_weights
    
    def get_average_acc(self):
        sum_acc = 0
        for i in self.clients:
             with open(i + '/model.json') as json_file: #with open(i.local_dir + 'model.json') as json_file:
                data = json.load(json_file)    
                local_acc = np.array([data['Accuracy']])
                sum_acc += local_acc
        return sum_acc/self.num_of_clients
    
    def prepare_weights(self):
        weights = []
        history = []
        with open(self.server_dir + '/standard_model_'+self.TrainingModel+'.json') as json_file:
            standard_weights = json.load(json_file)
            standard_weights = np.array([np.array(i) for i in standard_weights['weights']])
        
        average_acc = 0 #  self.get_average_acc()
        n_selected_clients = 0
        selected_clients = {}
        for i in self.clients:
            if ((self.common_features_nodes is None) and i == 'Common'):
                continue
                
            with open(i + '/model.json') as json_file: #with open(i.local_dir + 'model.json') as json_file:
                data = json.load(json_file)    
                local_weights = np.array([np.array(i) for i in data['weights']])
                history.append(data['performance'])
                #local_acc = np.array([data['Accuracy']])
                if(False): #local_acc < average_acc
                    selected_clients[i] = 0
                    continue
                
                weights.append(self.Equalize_weights(standard_weights,local_weights))
                n_selected_clients += 1
                selected_clients[i] = 1
        # f =  open('selected_Clients.txt', '+a');
        # f.write(selected_clients)
        # f.write('\n')
        # f.close()
        self.aggregate_and_save_history(history)
        return weights, n_selected_clients

    def load_client_weights(self):  
        weights = []
        for i in self.clients:
            with open(i.local_dir + 'model.json') as json_file:
                data = json.load(json_file)    
                weights.append(data['weights'])
        return weights
    
    def average_weights(self, weights):
        return np.mean(weights, axis=0)
        
    def aggregate_and_save_history(self, history, ag_type = 'server'):
        ag_dict = {}
        for h_dict in history:
            for k in h_dict:
                if k not in ag_dict.keys():
                    ag_dict[k] = h_dict[k]
                else:
                    ag_dict[k] = [x + y for x, y in zip(ag_dict[k], h_dict[k])]
        for i in ag_dict:
            ag_dict[i] = [x/len(history) for x in ag_dict[i]]
        df = pd.DataFrame(ag_dict, columns = ag_dict.keys())
        header = False
        mode = 'a'
        if(self.current_server_epoch == 1):
            header = True
            mode = 'w'
        if(ag_type == 'server'):
            df.to_csv(self.server_dir+'/performance.csv', mode= mode,header=header)
        else:
            df.to_csv('Common/performance.csv', mode=mode, header=header)
        return ag_dict
        
    def aggregate_commons(self):
        weights = []
        history = []
        for i in self.common_features_nodes:
            with open(i + '/model.json') as json_file: #with open(i.local_dir + 'model.json') as json_file:
                    data = json.load(json_file)    
                    local_weights = np.array([np.array(i) for i in data['weights']])
                    history.append(data['performance'])
                    weights.append(local_weights)
                    #print(local_weights[0].shape)
        ag_dict = self.aggregate_and_save_history(history,'Common')
        aggregated_weights = []
        for i in range(0, len(weights[0])):
            t = []
            for c in range(0, len(self.common_features_nodes)): #self.num_of_clients
                t.append(weights[c][i])
            aggregated_weights.append((self.average_weights(t)))
       
        self.save_model_json(ag_dict, aggregated_weights, w_type = 'Common')
        return aggregated_weights
    
    
    def aggregate_weights(self):
        #weights = self.load_client_weights()
        if not (self.common_features_nodes is None):
            self.aggregate_commons()
        
        weights, n_selected_clients  = self.prepare_weights()

    
        aggregated_weights = []
        for i in range(0, len(weights[0])):
            t = []
            for c in range(0, len(self.clients)-1): #self.num_of_clients
                t.append(weights[c][i])
            aggregated_weights.append((self.average_weights(t)))
        #self.save_model_json(aggregated_weights)
        return aggregated_weights
    
    def get_weighted_model(self,weights):
        model = Sequential()
        model.add(Dense(12, input_dim=2, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.set_weights(weights)
        model.compile(loss='mean_squared_error', optimizer='SGD',metrics=['accuracy'])
        return model
    
    def save_model_json(self, performance, weights,w_type = None):
        payloads = {'performance': performance ,'weights': weights}
        payloads = json.dumps(payloads, cls=NumpyEncoder)
        if (w_type is None):
            with open(self.server_dir+'/model.json', 'w') as server_dir:
                server_dir.write(payloads)
            # print('NonIID weights')
        else:
            with open('Common/model.json', '+w') as server_dir:
                server_dir.write(payloads)
            # print('Common Weights')
    
    def send_initial_model():
        model = Sequential()
        model.add(Dense(12, input_dim=2, activation='relu'))
        model.add(LSTM(4, input_shape=(2, 1)))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='SGD',metrics=['accuracy'])
        return model
        
    def start(self):
        # if(self.Initiate):
        #     self.send_initial_model()
        #     self.start = True
        updated_weights = self.aggregate_weights()
        self.save_model_json(1,updated_weights)
        # model = self.get_weighted_model(updated_weights)
        # model.save(self.server_dir)
        print('---- Model Saved By server ----')
    
    def check_client_status(self):
        f =  open('monitor_clients.txt', 'r');
        temp = [x for x in f.read().split(',')]
        if(len(temp) == self.num_of_clients+1):
            f.close()
            return 1
        else:
            f.close()
            return 0
    def _timer(self,s,e,file_name):
        t = e-s
        f =  open(self.server_dir +"/"+ file_name + '.csv', '+a');
        f.write(str(t) +',')
        f.close()
        
    def start_parallel(self):
        server_epochs = 20    # 1. Chnage This if want to continue
        waiting_time_start = time.time()
        while(True):
            print('---- Checking Status ------')
            self.current_server_epoch = server_epochs
            status = self.check_client_status()
            if(status == 1):
                waiting_time_end = time.time()
                self._timer(waiting_time_start,waiting_time_end,'waiting_time')
                train_time_start = time.time()
                print('--------------Server Round '+str(server_epochs) +'-------------------')
                self.start()
                train_time_end = time.time()
                self._timer(train_time_start,train_time_end,'aggregation_time')
                f =  open('monitor_clients.txt', 'w');
                f.write('')
                f =  open('monitor_server.txt', 'a');
                f.write(',' + str(server_epochs))
                f.close()
                server_epochs += 1
                waiting_time_start = time.time()
                #self.target.individual_model_test()
                
                
            if(self.server_epochs == server_epochs ):
                break
            print('---- Waiting ------' + str(time.time() - waiting_time_start) +' seconds')
            time.sleep(10)
                

s = server()
s.start_parallel()
# c1 = ThermalClient(1, 5, 'Italy')
# c2 = ThermalClient(2, 5,'Portugal')
# c3 = ThermalClient(1, 5, 'Philippines')
# s = server([c2,c3])

# s.start()

