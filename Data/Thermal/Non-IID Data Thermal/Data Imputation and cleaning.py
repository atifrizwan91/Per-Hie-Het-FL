# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:05:39 2022

@author: user
"""
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
drop_list = ['Unnamed: 0','Publication (Citation)','Data contributor','Heating strategy_building level',
                     'Year','Koppen climate classification','Climate','Building type','Database',
                     'City','Country','Outdoor monthly air temperature (¡C)','Thermal preference','Season','Cooling startegy_building level',
                     'Air movement preference','Humidity preference','Cooling startegy_operation mode for MM buildings']

countries = ['India', 'Italy', 'Pakistan', 'Philippines', 'Portugal', 'Singapore', 'Sweden', 'Thailand', 'Tunisia',  'UK' ]



def imputeData_KNN(data, country):
    cols = data.columns
    imputer = KNNImputer(n_neighbors=3, weights='uniform', metric='nan_euclidean')
    imputer.fit(data)
    Xtrans = imputer.transform(data)

    df_Imputed = pd.DataFrame(Xtrans, columns = cols)
    df_Imputed.to_csv(country+'_Imputed_Ashrae.csv')
    
def start_imputation():
    for i in countries:
        df = pd.read_csv(i+'_Ashrae.csv')
        print(i)
        drop = list(set(df.columns).intersection(drop_list))
        df = df.drop(drop, axis = 1)
        imputeData_KNN(df,i)

def data_info():
    info = {}
    for i in countries:
        df = pd.read_csv(i+'_Imputed_Ashrae.csv')
        info[i] = [len(df.index), len(df.columns)]
    df = pd.DataFrame(info)
    df.to_csv('data_info.csv')
# start_imputation()
data_info()
a = np.array([[2,3],[4,5],[6,7],[8,9],[0,1],[2,3],[43,45],[5,6],[7,4]])
print(np.split(a,2))
