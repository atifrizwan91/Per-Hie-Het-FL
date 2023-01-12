
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import math
# from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input
from pathlib import Path
from keras.layers import LSTM
from keras.layers import Activation 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Flatten, Conv1D, SpatialDropout1D, BatchNormalization
# from keras.layers import Embedding,Conv1D,LSTM,Input,TimeDistributed,,Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from keras.metrics import categorical_crossentropy
import glob
from sklearn.metrics import f1_score
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


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
    
class ThermalClient:
    def __init__(self, country):
         
        conf = self.get_configuration()
        
        #print(conf)
        #self.client_num = client_num
        self.country = country
        self.first_iteration = False   #Change this for continue
        
        self.local_dir = country+"/"
        self.server_dir = conf['server_dir'] #'./Server-Dir'
        self.data_dir = ''
        self.detect_change = 14 # 2. Chnage This if want to continue # set to 1 for start from begining
        self.local_epochs = conf['client_epochs']
        self.loss_history = []
        self.drop_list = ['Unnamed: 0','Publication (Citation)','Data contributor','Heating strategy_building level',
                     'Year','Koppen climate classification','Climate','Building type','Database',
                     'City','Country','Outdoor monthly air temperature (¡C)','Thermal preference','Season','Cooling startegy_building level',
                     'Air movement preference','Humidity preference','Thermal comfort','Thermal sensation','Clo','label']
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_data()
        #print(self.X_train.shape[1],self.X_train.shape[2])
        
        # Un comment
        # Path(""+self.country).mkdir(parents=True, exist_ok=True)
        # f = open(self.local_dir+ 'performance.csv', '+w')
        # f.close()
        # f = open(self.local_dir+ 'performance_after_round.csv', '+w')
        # f.close()
    
    def get_configuration(self):
        with open('config.json') as json_file:
             conf = json.load(json_file)
        return conf
    
    def get_data(self):
       # df = pd.read_csv("D:\\Projects\Federated Learning\\Datasets\\Fed Dataset Thermal (Transfer Learning Paper)\\"+self.country+".csv")
        df = pd.read_csv('Data/Correlation Based Data/'+self.country+'.csv')
        y = df['Thermal sensation']
        drop = list(set(df.columns).intersection(self.drop_list))
        X = df.drop(drop, axis = 1)
        #print(X.columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # df_X_train = pd.DataFrame(X_train)
        # df_X_test = pd.DataFrame(X_test)
        # df_y_train = pd.DataFrame(y_train)
        # df_y_test = pd.DataFrame(y_test)
        
        # df_X_train.to_csv('df_X_train.csv')
        # df_X_test.to_csv('df_X_test.csv')
        # df_y_train.to_csv('df_y_train.csv')
        # df_y_test.to_csv('df_y_test.csv')
        
        y_train=to_categorical(y_train,num_classes=5)
        y_test=to_categorical(y_test,num_classes=5)
        
        X_train = np.asarray(X_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        
        #for LSTM CNN model
        X_train=np.array(X_train).reshape(X_train.shape[0],X_train.shape[1],-1)
        X_test=np.array(X_test).reshape(X_test.shape[0],X_test.shape[1],-1)
        #For DNN Model
        #y_train = np.reshape(y_train, (len(y_train), len(y_train[0])))
        
        return X_train, X_test, y_train, y_test
        
    def get_thermal_model(self):
        model=Sequential()
        #print(self.X_train.shape[2],"----------------")
        #model.add(Embedding(101,256,input_length=8,))
        model.add(Conv1D(filters=128,kernel_size=5,padding='same',input_shape=(self.X_train.shape[1],self.X_train.shape[2])))
        model.add(LSTM(256,return_sequences=True))
        model.add(LSTM(256,return_sequences=True))
        model.add(Flatten())
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(16,activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(5,activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001,beta_1=0.99,beta_2=0.999),metrics=['accuracy'],weighted_metrics=['accuracy'])
        return model
    
    def build_model(self):
      MLP_model=Sequential()
      #np.reshape(a,(len(a),1,len(a[0])))
      
      MLP_model.add(Dense(1024,activation='relu',input_shape = (len(list(self.X_train[0])),)))
      
      MLP_model.add(Dense(512,activation='relu',kernel_initializer='glorot_uniform'))
      MLP_model.add(Dense(256,activation='relu',kernel_initializer='glorot_uniform'))
      MLP_model.add(Dense(128,activation='relu',kernel_initializer='glorot_uniform'))
      MLP_model.add(Dense(64,activation='relu',kernel_initializer='glorot_uniform'))
      MLP_model.add(Dense(32,activation='relu',kernel_initializer='glorot_uniform'))
      MLP_model.add(Dense(16,activation='relu',kernel_initializer='glorot_uniform'))
      MLP_model.add(Dense(8,activation='relu',kernel_initializer='glorot_uniform'))
      # MLP_model.add(BatchNormalization())
      MLP_model.add(Dense(5,activation='softmax'))
      MLP_model.compile(optimizer=Adam(lr=0.001),metrics=['accuracy'],loss='categorical_crossentropy',weighted_metrics=['accuracy'])
      # checkpoint_filepath = '/tmp/checkpoint'
      # es=ModelCheckpoint(filepath=checkpoint_filepath,monitor='val_accuracy',save_best_only=True,mode='max',save_weights_only=True)
      # MLP_model.fit(X_train,y_train,epochs=150,validation_split=0.2,batch_size=64,callbacks=[es],class_weight=weight_dicts)
      # MLP_model.load_weights(checkpoint_filepath)
      return MLP_model
    
    def build_model_LSTM_CNN(self):
      # self.y_train=to_categorical(self.y_train,num_classes=5)
      model=Sequential()
      model.add(Conv1D(filters=128,kernel_size=5,padding='same',input_shape=(self.X_train.shape[1],self.X_train.shape[2])))
      model.add(SpatialDropout1D(0.1))
      model.add(LSTM(256,return_sequences=True))
      model.add(LSTM(256,return_sequences=True))
      model.add(Flatten())
      model.add(Dense(64,activation='relu'))
      model.add(Dense(32,activation='relu'))
      model.add(Dense(16,activation='relu'))
      model.add(Dense(8,activation='relu'))
      model.add(Dense(5,activation='softmax'))
      model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'],weighted_metrics=['accuracy'])
      # checkpoint_filepath = '/tmp/checkpoint'
      # es=ModelCheckpoint(filepath=checkpoint_filepath,monitor='val_accuracy',save_best_only=True,mode='max',save_weights_only=True)
      # model.fit(X_train,y_train,epochs=100,validation_split=0.2,batch_size=64,callbacks=[es],class_weight=weight_dicts)
      # model.load_weights(checkpoint_filepath)
      return model
    
    def build_model_LSTM(self):
      # self.y_train=np.asarray(self.y_train ,dtype=int)
      self.y_train=to_categorical(self.y_train,num_classes=5)
      self.X_train=self.X_train.reshape(self.X_train.shape[0],self.X_train.shape[1],-1)
      model=Sequential()
      model.add(LSTM(256,return_sequences=True,input_shape=(self.X_train.shape[1],self.X_train.shape[2])))
      model.add(LSTM(256,return_sequences=True))
      model.add(LSTM(256,return_sequences=True))
      model.add(Flatten())
      model.add(Dense(64,activation='relu'))
      model.add(Dense(32,activation='relu'))
      model.add(Dense(16,activation='relu'))
      model.add(Dense(8,activation='relu'))
      model.add(Dense(5,activation='softmax'))
      model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'],weighted_metrics=['accuracy'])
      # checkpoint_filepath = '/tmp/checkpoint'
      # es=ModelCheckpoint(filepath=checkpoint_filepath,monitor='val_accuracy',save_best_only=True,mode='max',save_weights_only=True)
      # model.fit(X_train,y_train,epochs=150,validation_split=0.2,batch_size=64,callbacks=[es],class_weight=weight_dicts)
      # model.load_weights(checkpoint_filepath)
      return model
  
    def get_mmmm(self):
        model=Sequential()
        model.add(Dense(64, input_dim = 6, activation='relu'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001,beta_1=0.99,beta_2=0.999),metrics=['accuracy'],weighted_metrics=['accuracy'])
        return model
    
    def get_weighted_model1(self,weights):
        model = self.get_thermal_model() #self.build_model() #self.get_thermal_model()
        model.set_weights(weights)
        return model
    
    def train_model(self):
        model = self.get_thermal_model() # self.build_model() #self.get_thermal_model()
        history = model.fit(self.X_train, self.y_train, validation_data=(self.X_test,self.y_test), epochs=self.local_epochs, batch_size=1, verbose=2)
        weights = model.get_weights()
        #print(history.history)
        accuracy = history.history['val_accuracy']
        
        
        loss = history.history['val_loss']
        self.save_history_full(history.history)
        self.save_history(loss,accuracy, '+w')
        loss = loss[-1]
        accuracy = accuracy[-1]
        self.loss_history.append(loss)
        #ev = {'loss': loss, 'accuracy': accuracy}
        self.first_iteration = False
        self.save_model_json(history.history, weights)
        return loss, accuracy, weights
    
    def save_model_json(self,performance, weights):
        payloads = {'performance': performance,  'weights': weights}
        payloads = json.dumps(payloads, cls=NumpyEncoder)
        with open(self.local_dir+'model.json', 'w') as local_dir:
            local_dir.write(payloads)

    def save_model(self, model):
        model.save(self.local_dir)

    def train_n_save_model(self):
        loss, accuracy, weights = self.train_model()
        
        #self.save_model(model)

    def save_history(self,loss, accuracy,status):
        fo = open(self.local_dir+'loss.csv', status)
        for i,j in zip(loss,accuracy):
            fo.write(str(i)+',' + str(j) +'\n')
        fo.close()
        
        
    def save_history_full(self,performance):
        df = pd.DataFrame(performance)
        if(self.first_iteration):
            df.to_csv(self.local_dir+'performance.csv', mode='w',header=True)
        else:
            df.to_csv(self.local_dir+'performance.csv', mode='a', header=False)
            
    
    def test_received_model(self,model):
        train_p = model.predict(self.X_train)
        test_p = model.predict(self.X_test)
        cce = tf.keras.losses.CategoricalCrossentropy()
        acc = tf.keras.metrics.Accuracy()
        
        #Train data performance
        train_loss = cce(self.y_train, train_p).numpy()     
        acc.update_state(self.y_train, train_p)
        #tain_accuracy = acc.result().numpy()
        
        cat_train_p = self.my_to_catagorical(train_p)
        tain_accuracy = accuracy_score(self.y_train, cat_train_p)
        report = classification_report(self.y_train, cat_train_p,output_dict=True)
        precision = [[c,p['precision']] for [c,p] in report.items()]
        train_classwise_precision = precision[0:5]
        train_avg_precision = precision[8]
        f1_train = f1_score(self.y_train, cat_train_p, average='weighted')
        
        
         #Test performance
        test_loss = cce(self.y_test, test_p).numpy()
        acc.update_state(self.y_test, test_p)
        # test_accuracy = acc.result().numpy()
        cat_test_p = self.my_to_catagorical(test_p)
        test_accuracy = accuracy_score(self.y_test, cat_test_p)
        report = classification_report(self.y_test, cat_test_p,output_dict=True)
        precision = [[c,p['precision']] for [c,p] in report.items()]
        test_classwise_precision = precision[0:5]
        test_avg_precision = precision[8]
        f1_test = f1_score(self.y_test, cat_test_p, average='weighted')
        
        df = pd.DataFrame([[f1_train,f1_test,train_loss ,tain_accuracy,train_avg_precision[1] ,test_loss,test_accuracy,test_avg_precision[1],
                            train_classwise_precision[0][1],train_classwise_precision[1][1],train_classwise_precision[2][1],train_classwise_precision[3][1],train_classwise_precision[4][1],
                            test_classwise_precision[0][1],test_classwise_precision[1][1],test_classwise_precision[2][1],test_classwise_precision[3][1],test_classwise_precision[4][1]]], 
                          columns = ['f1_train','f1_test','TrainLoss','TrainAcc','TrainPrecision','TestLoss','TestAcc','TestPrecision','TrP1','TrP2','TrP3','TrP4','TrP5','TsP1','TsP2','TsP3','TsP4','TsP5'])
        df.to_csv(self.local_dir+'performance_after_round.csv', mode='a', header=False)
    
    
    def finetune_model(self,model):
        history = model.fit(self.X_train, self.y_train, validation_data=(self.X_test,self.y_test), epochs=self.local_epochs, batch_size=1, verbose=2)
        weights = model.get_weights() 
        accuracy = history.history['val_accuracy']
        loss = history.history['val_loss']
        self.save_history(loss, accuracy, 'a')
        self.save_history_full(history.history)
        loss = loss[-1]
        accuracy = accuracy[-1]
        #self.loss_history.append(loss)
        ev = {'loss': loss, 'accuracy': accuracy}
        self.save_model_json(history.history, weights)
        return loss, accuracy, weights
    
    def prepare_weights(self, server_weights):
        with open(self.local_dir + '/model.json') as json_file:
            local_weights = json.load(json_file)
        local_weights = np.array([np.array(i) for i in local_weights['weights']])
        for i in range(0,len(local_weights)):
            if (local_weights[i].shape != server_weights[i].shape):
                temp = server_weights[i]
                #print(temp.shape)
                for ii in range(len(server_weights[i]), len(local_weights[i])):
                     temp = np.append(temp, [local_weights[i][ii]], axis = 0)
                server_weights[i] = temp
        return server_weights
    
    def my_to_catagorical(self, y_predict):
        updated_y_predict = []
        for p in y_predict:
            n = max(p)
            for i in range(0,len(p)):
               if n == p[i]:
                   p[i] = 1
               else:
                   p[i] = 0
            updated_y_predict.append(p)
        return updated_y_predict
    
    def individual_model_test(self):
        with open(self.server_dir + '/model.json') as json_file:
            data = json.load(json_file)
        weights = np.array([np.array(i) for i in data['weights']])
        weights = self.prepare_weights(weights)
        model = self.get_weighted_model1(np.asarray(weights))
        
        train_p = model.predict(self.X_train)
        test_p = model.predict(self.X_test)
        cce = tf.keras.losses.CategoricalCrossentropy()
        acc = tf.keras.metrics.Accuracy()
        
        #Train data performance
        train_loss = cce(self.y_train, train_p).numpy()     
        acc.update_state(self.y_train, train_p)
        
        # tain_accuracy = acc.result().numpy()
        
        cat_train_p = self.my_to_catagorical(train_p)
        tain_accuracy = accuracy_score(self.y_train, cat_train_p)
        report = classification_report(self.y_train, cat_train_p,output_dict=True)
        precision = [[c,p['precision']] for [c,p] in report.items()]
        train_classwise_precision = precision[0:5]
        train_avg_precision = precision[8]
        
        
        
         #Test performance
        test_loss = cce(self.y_test, test_p).numpy() 
        acc.update_state(self.y_test, test_p)
        # test_accuracy = acc.result().numpy()
        
        cat_test_p = self.my_to_catagorical(test_p)
        test_accuracy = accuracy_score(self.y_test, cat_test_p)
        report = classification_report(self.y_test, cat_test_p,output_dict=True)
        precision = [[c,p['precision']] for [c,p] in report.items()]
        test_classwise_precision = precision[0:5]
        test_avg_precision = precision[8]
        
        
        
        
        df = pd.DataFrame([[train_loss ,tain_accuracy,train_avg_precision[1] ,test_loss,test_accuracy,test_avg_precision[1],
                            train_classwise_precision[0][1],train_classwise_precision[1][1],train_classwise_precision[2][1],train_classwise_precision[3][1],train_classwise_precision[4][1],
                            test_classwise_precision[0][1],test_classwise_precision[1][1],test_classwise_precision[2][1],test_classwise_precision[3][1],test_classwise_precision[4][1]]], 
                          columns = ['TrainLoss','TrainAcc','TrainPrecision','TestLoss','TestAcc','TestPrecision','TrP1','TrP2','TrP3','TrP4','TrP5','TsP1','TsP2','TsP3','TsP4','TsP5'])
        df.to_csv(self.local_dir+'Last Performance.csv', mode='+w', header=True)
    
    def start(self):
        # list_of_files = glob.glob(self.server_dir+'/*')
        # latest_file = max(list_of_files, key=os.path.getctime)
        with open(self.server_dir + '/model.json') as json_file:
            data = json.load(json_file)
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        weights = np.array([np.array(i) for i in data['weights']], dtype=object)
        weights = self.prepare_weights(weights)
        model = self.get_weighted_model1(np.asarray(weights))
        self.test_received_model(model)
        loss, accuracy, weights = self.finetune_model(model)
        
    def _timer(self,s,e,file_name):
        t = e-s
        f =  open(self.local_dir + file_name + '.csv', '+a')
        f.write(str(t) +',')
        f.close()
        
    def check_server_status(self):
        f = open('monitor_server.txt', 'r');
        temp = [int(x) for x in f.read().split(',')]
        
        if(self.detect_change == temp[-1]):
            self.detect_change += 1
            return 1
        else:
            return 0
        
    def start_parallel(self):
        # Uncomment All for start from specific round
        # train_time_start = time.time()
        # self.train_n_save_model()
        # train_time_end = time.time()
        # self._timer(train_time_start,train_time_end,'training_time')
        # f =  open('monitor_clients.txt', 'a');
        # f.write(self.country + ",")
        # f.close()
        waiting_time_start = time.time()
        while(True):
            print('---- Waiting ------'+ str(time.time() - waiting_time_start) +' seconds')
            time.sleep(60)
            status = self.check_server_status()
            if(status == 1):
                waiting_time_end = time.time()
                self._timer(waiting_time_start,waiting_time_end,'waiting_time')
                train_time_start = time.time()
                self.start()
                train_time_end = time.time()
                self._timer(train_time_start,train_time_end,'training_time')
                f =  open('monitor_clients.txt', 'a')
                f.write(self.country + ",")
                f.close()
                waiting_time_start = time.time()
                
                
            
            
            
        #self.update_server_status(1)
        
    # def load_model(self):
    #     model = keras.models.load_model(self.server_dir) 
        
        # model = self.finetune_model(model) 
        # self.save_model(model)
# s = 0

# while(True):
#     time.sleep(2000)


# c = ThermalClient('India')
# print(c.X_train.columns)
# c.individual_model_test()
# c.start_parallel()

# print(c.y_train[1].shape)
# print(len(c.y_train))


# c.y_train = np.reshape(c.y_train, (len(c.y_train), len(c.y_train[0])))
# print(c.y_train[0].shape)
# c.train_n_save_model()
# c.start()
# print(c.X_train[0])
