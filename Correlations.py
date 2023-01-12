# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 23:10:25 2022

@author: user
"""

import pandas as pd

labels = ['India', 'Italy', 'Pakistan', 'Philippines', 'Portugal', 'Singapore', 'Sweden', 'Thailand', 'Tunisia','UK',"Scales","Medium_US"]

# df = pd.read_csv('Data/Thermal Proposed/data_1.csv')
drop_list = ['Unnamed: 0','Publication (Citation)','Data contributor','Heating strategy_building level',
                     'Year','Koppen climate classification','Climate','Building type','Database',
                     'City','Country','Outdoor monthly air temperature (¡C)','Thermal preference','Season','Cooling startegy_building level',
                     'Air movement preference','Humidity preference','Thermal comfort','Thermal sensation']


print()

for i in labels:
    df = pd.read_csv('Data/Thermal Proposed/'+i+'.csv')
    corr = df.corrwith(df['Thermal sensation'])
    corr.to_csv('Correlations/'+i+'_corr.csv')
    
    