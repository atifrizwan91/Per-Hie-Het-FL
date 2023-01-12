# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:54:19 2022

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 18:55:56 2022

@author: user
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Loading the Raw datafile
# Modify pathstr accordingly
path_str = 'ashrae_db2.01.csv'
data_ash=pd.read_csv(path_str,encoding='ISO-8859-1')

def check_percentage(data):
    
    NA_count = data.isna().sum()
    total = len(data)
    return (NA_count/total) * 100
    

def process_data(data_ash,country):
    data_ash = data_ash[data_ash['Country'] == country]
    for i in data_ash.columns:
        p = check_percentage(data_ash[i])
        if (p>30):
            data_ash = data_ash.drop(i, axis = 1)
    return data_ash
            
def fun(str1):
  if(str1=='Female'):
    return(2.0)
  elif(str1=='Male'):
    return(1.0)

countries = data_ash['Country'].unique()
for i in countries: 
    data = process_data(data_ash,i)
    data['Thermal sensation'] = data['Thermal sensation'].apply(lambda x: -2 if x <= -2 else x)
    data['Thermal sensation'] = data['Thermal sensation'].apply(lambda x: 2 if x >= 2 else x)
    data['Thermal sensation'] = data['Thermal sensation'].apply(lambda x: np.round(x))
    data_ash['Sex']=data_ash['Sex'].apply(lambda x: fun(x))
    data.to_csv(i+'_Ashrae.csv')
    
def missing_matrix(data_ash,country):
    data_ash = data_ash[data_ash['Country'] == country]
    percentage = []
    for i in data_ash.columns:
        p = check_percentage(data_ash[i])
        percentage.append(p)
    return percentage

def plot_missing_matrix(matrix,countries, cols):
    matrix = matrix.values()
    matrix_r = []
    for i in matrix:
        temp = []
        for ii in i:
            #t = ii*100
            t = int(ii)
            temp.append(t)
        matrix_r.append(temp)
    print(matrix_r)
    df_cm = pd.DataFrame(matrix_r, index = countries, columns = cols)
    fig = plt.figure(figsize = (40,7))
    sns.heatmap(df_cm, annot=True, fmt=".0f")
    plt.tight_layout()
    fig.savefig('missingplot.pdf',bbox_inches='tight')
    
#make a matrix of % of mssing values
missing_p = {}
for i in countries:
    percentage = missing_matrix(data_ash,i)
    missing_p[i] = percentage

print(missing_p.values())
plot_missing_matrix(missing_p,countries,data_ash.columns)
df_missing_p = pd.DataFrame(missing_p)


def locate_column(_new_cols):
    cols = list(data_ash.columns)
    cols.append('Unnamed: 0')
    f_list = [0]*len(cols)
    for i in _new_cols:
        index = cols.index(i)
        f_list[index] = i
    return f_list
def save_columns(countries):
    cols = []
    for i in countries:
        df_c = pd.read_csv(i+'_Ashrae.csv')
        f_list = locate_column(df_c.columns)
        f_list.insert(0,i)
        cols.append(f_list)
    df_c = pd.DataFrame(cols)
    
    df_c.to_csv('Data Common Columns.csv')
    
save_columns(countries)  


def set_columns(country):
    df_cou = pd.read_csv(country+"_Ashrae.csv")
    
    print(df_cou.head())
    df_temp = pd.DataFrame()
    seq = ['Country','PMV','PPD','SET','Clo','Met','Air temperature (Â¡C)','Relative humidity (%)','Air velocity (m/s)']
    for i in seq:
        print(i)
        if(i in df_cou.columns):
            print('True')
            df_temp[i] = df_cou[i]
            df_cou = df_cou.drop(i,axis = 1)

    df_temp = pd.concat([df_temp, df_cou], axis=1)
    df_temp.to_csv(country +'_Ashrae.csv')

for i in countries: 
      set_columns(i)
      print(i)
      

print('PMV' in countries)
print(countries)
#Removing all the other data not having AC as thier Cooling stratergy
data1=data_ash[data_ash['Cooling startegy_building level']=='Air Conditioned']
# data_ash=pd.concat([data1,data2,data3],axis=0)

#Making the range from [-3,3] to[2,2]
data_ash['Thermal sensation'] = data_ash['Thermal sensation'].apply(lambda x: -2 if x <= -2 else x)
data_ash['Thermal sensation'] = data_ash['Thermal sensation'].apply(lambda x: 2 if x >= 2 else x)
#Rounding off the values to make it categorical in nature 
data_ash['Thermal sensation'] = data_ash['Thermal sensation'].apply(lambda x: np.round(x))


data_ash=data_ash.drop_duplicates()



    


data_ash.columns

data_ash=data_ash[data_ash['Air velocity (m/s)']>0]
data_ash=data_ash[data_ash['Air temperature (¡C)']>0]
data_ash=data_ash[data_ash['Radiant temperature (¡C)']>0]

data_ash['Clo'].describe()

data_ash=data_ash.fillna(data_ash.median())

data_ash=data_ash.drop_duplicates()
data_ash=data_ash.dropna()



from sklearn.cluster import DBSCAN

# def DBSCAN_outlier_detection(data):
#   outlier_detection=DBSCAN(min_samples=5,eps=3)
#   clusters=outlier_detection.fit_predict(data)
#   data['Clusters']=clusters
#   data=data[data['Clusters']!=-1]
#   data=data.drop(['Clusters'],axis=1)
#   return(data)

# data_ash=DBSCAN_outlier_detection(data_ash)

data_ash.to_csv('Ashrae.csv')