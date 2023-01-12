# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:34:04 2022

@author: user
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import math
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
from sklearn.decomposition import PCA

def get_configuration():
        with open('config.json') as json_file:
             conf = json.load(json_file)
        return conf

conf = get_configuration()
clients = conf['common_features_nodes']
clients.extend(conf['clients'])
clients.remove('Common')

clients = ["data_0","data_1","data_2","data_3","data_4","data_5","data_6","data_7","data_8","data_9"]

def get_data():
    all_data = {}
    for i in clients:
        df = pd.read_csv('Data/Synthetic/'+i+'.csv')
        all_data[i] = df
    return all_data

def apply_PCA(data,components = 6):
    pca = PCA(n_components = components)
    y = data['Thermal sensation']
    X = data.drop('Thermal sensation', axis = 1)
    com = pca.fit_transform(X)
   
    df = pd.DataFrame(com[:,0], columns=['PCA1'])
    df['PCA2'] = com[:,1]
    df['Thermal sensation'] = y
    return df

all_data = get_data()

def plot_components():
    for c in clients:
        df = apply_PCA(all_data[c])
        fig = plt.figure()
        plt.scatter(df['PCA1'], df['PCA2'],c=df['Thermal sensation'], alpha=0.9)
        plt.axis('equal');
plot_components()
    