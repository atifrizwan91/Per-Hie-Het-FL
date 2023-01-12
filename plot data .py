# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:27:36 2022

@author: mcl
"""

import pandas as pd
import matplotlib.pyplot as plt

countries = ["AF","AS","EU","NA","OC","SA","Common","Medium_US","Scales"]

for i in countries:
    df = pd.read_csv(i+'/performance.csv')
    fig = plt.figure()
    x_axis = [x for x in range(0, len(df['loss']))]
    plt.plot(x_axis, df['loss'])
    try:
        df1 = pd.read_csv(i+'/performance_after_round.csv')
        x_axis = [x*5 for x in range(0, len(df1['mae']))]
        plt.plot(x_axis, df1['mae'])
    except:
        print('No')
    
    



