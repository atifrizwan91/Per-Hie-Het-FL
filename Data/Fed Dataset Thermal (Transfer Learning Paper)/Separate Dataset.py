# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 23:01:18 2022

@author: user
"""

import pandas as pd

df = pd.read_csv('D:\\Projects\\Federated Learning\\Datasets\\Fed Dataset Thermal (Transfer Learning Paper)\\Ashrae.csv')
countries = ['India', 'Italy', 'Pakistan', 'Philippines', 'Portugal', 'Singapore', 'Sweden', 'Thailand', 'Tunisia',  'UK' ]

countries = df['Country'].unique()

for i in countries:
    print("\"" +i+ "\"", end = ",")

for i in countries:
    df1 = df[df['Country'] == i]
    df1.to_csv('D:\\Projects\\Federated Learning\\Datasets\\Fed Dataset Thermal (Transfer Learning Paper)\\'+i+'.csv')

    