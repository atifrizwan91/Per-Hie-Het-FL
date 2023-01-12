# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:08:59 2022

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

data = pd.read_csv('Data\\Thermal\\All Data\\ashrae_db2.01.csv', encoding='latin1')

cols = ['PMV','PPD','SET','Clo','Met','Air temperature (¡C)','Relative humidity (%)','Air velocity (m/s)','Outdoor monthly air temperature (¡C)','Thermal sensation']

df = data[cols]
df = df.dropna()

classes = ['-2','-1','0','1','2']

class_wise = {}

for i in classes:
    class_wise[i] = df[df['Thermal sensation'] == int(i)]

n = 10
df = pd.DataFrame(columns=['Air temperature (¡C)','Relative humidity (%)','Air velocity (m/s)','Outdoor monthly air temperature (¡C)','Thermal sensation','PMV','PPD','SET','Clo','Met'])
final_data = {}
#initilize with empty dataframe
for i in range(0,n):
    final_data[i] = df
    
for i in classes:
    total = len(class_wise[i].index)
    batch = int(total/n)
    start = 0
    end = batch
    for j in range(0,n):
        df_new = class_wise[i][start:end]
        
        df = final_data[j]
        df = pd.concat([df,df_new])
        final_data[j] = df
        start = end
        end += batch


for i in range(0,n):
    df = final_data[i]
    df.to_csv('Data/Synthetic/data_'+str(i)+'.csv')

