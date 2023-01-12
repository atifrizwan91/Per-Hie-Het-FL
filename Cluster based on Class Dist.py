# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 00:03:56 2022

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
from sklearn.cluster import KMeans
import pandas as pd

clients = ['India','Italy',"Pakistan","Philippines","Portugal", "Singapore","Sweden","Thailand","Tunisia","UK"]
classes = [-2,-1,0,1,2]
class_dist = {}
for i in clients:
    df = pd.read_csv('Data/Thermal Proposed/'+i+'.csv')
    temp = []
    for j in classes:
        t = df[df['Thermal sensation'] == j]
        temp.append(len(t.index))
    class_dist[i] = temp

df = pd.DataFrame(class_dist)
df_t = df.T
kmeans = KMeans(n_clusters=2, random_state=2).fit(df_t)
print(kmeans.labels_)
