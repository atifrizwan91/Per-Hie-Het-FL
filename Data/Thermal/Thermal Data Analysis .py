# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 00:01:39 2022

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('Thermal C4.csv')
y = df['Thermal sensation']

unique, counts = np.unique(y, return_counts=True)
# plt.matshow(df.corr())
# plt.show()

def class_dist():
    fig = plt.figure()
    plt.bar(unique, counts)
    plt.show()
    
def hist(df):
    x1 = df.loc[df['Thermal sensation']==0, 'Air temperature (¡C)']
    x2 = df.loc[df['Thermal sensation']==1, 'Air temperature (¡C)']
    x3 = df.loc[df['Thermal sensation']==2, 'Air temperature (¡C)']

    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
    
    plt.figure(figsize=(10,7), dpi= 80)
    sns.distplot(x1, color="dodgerblue", label="Compact", **kwargs)
    sns.distplot(x2, color="orange", label="SUV", **kwargs)
    sns.distplot(x3, color="deeppink", label="minivan", **kwargs)
    plt.xlim(50,75)
    plt.legend();
    
hist(df)