# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:35:53 2022

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

# classes = ['-2','-1','0','1','2']
classes = ['0','1','2','3','4']
target = 'label'


def get_configuration():
        with open('config.json') as json_file:
             conf = json.load(json_file)
        return conf

conf = get_configuration()
clients = ['India', 'Italy', 'Pakistan', 'Philippines', 'Portugal', 'Singapore', 'Sweden', 'Thailand', 'Tunisia','UK',"Medium_US","Scales"] #conf['common_features_nodes']
clients = ['syn_data_0','syn_data_1',"syn_data_2","syn_data_3","syn_data_4", "syn_data_5","syn_data_6","syn_data_7","syn_data_8","syn_data_9"]
# clients.extend(conf['clients'])
# clients.remove('Common')

def get_data():
    all_data = {}
    for i in clients:
        # df = pd.read_csv('Data/Thermal Proposed/'+i+'.csv')
        df = pd.read_csv('Data/Synthetic/'+i+'.csv')
        all_data[i] = df
    return all_data

def get_class_distribution(label, data):
    distribution = Counter(data[label])
    return distribution

def get_all_distribution(data): # data is dictionary of all data
    all_distributions = {}
    for key in data:
        distribution = get_class_distribution(target, data[key])
        all_distributions[key] = distribution
    return all_distributions

def compute_correlations(label, data):
    c = data.corrwith(data[label])
    print(c)
    return c


def class_wiese_distribution(all_distributions):
    
    class_distribution = {classes[0]:[],classes[1]:[],classes[2]:[],classes[3]:[],classes[4]:[]}
    for key in all_distributions:
        for i in all_distributions[key]:
            i = int(i)
            class_distribution[str(i)].append(all_distributions[key][i])
    return class_distribution
    
all_data = get_data()

# compute_correlations('Thermal sensation',all_data['NA'])

all_distributions = get_all_distribution(all_data)

class_wise = class_wiese_distribution(all_distributions)

def plot_barcharts():
    r = [0,1,2,3,4,5,6,7,8,9,10,11]
    r = [0,1,2,3,4,5,6,7,8,9]
    #raw_data = {'greenBars': [20, 1.5, 7, 10, 5], 'orangeBars': [5, 15, 5, 10, 15],'blueBars': [2, 15, 18, 5, 10]}
    
    df = pd.DataFrame(class_wise)
    df = pd.read_csv('Syntehtic Data Distribution.csv')
    # From raw value to percentage
    totals = [i+j+k+l+m for i,j,k,l,m in zip(df[classes[0]], df[classes[1]], df[classes[2]],df[classes[3]],df[classes[4]])]
    
    m_two = [i / j * 100 for i,j in zip(df[classes[0]], totals)]
    m_one = [i / j * 100 for i,j in zip(df[classes[1]], totals)]
    zero = [i / j * 100 for i,j in zip(df[classes[2]], totals)]
    one = [i / j * 100 for i,j in zip(df[classes[3]], totals)]
    two = [i / j * 100 for i,j in zip(df[classes[4]], totals)]
    
    
    # m_two = df['-2']
    # m_one = df['-1']
    # zero = df['0']
    # one = df['1']
    # two = df['2']
    
    # plot
    barWidth = 0.95
    # names = ("India","Italy","Pakistan","Philippines","Portugal","Singapore","Sweden","Thailand","Tunisia","UK","Medium US","Scales")
    names = ("Client 1","Client 2","Client 3","Client 4","Client 5","Client 6","Client 7","Client 8","Client 9","Client 10")
    # Create green Bars
    fig = plt.figure()
    plt.bar(r, m_two, color='#BCE29E', edgecolor='white', width=barWidth, label = 'Class '+classes[0])
    plt.bar(r, m_one, bottom=m_two, color='#f9bc86', edgecolor='white', width=barWidth,label = 'Class '+classes[1])
    plt.bar(r, zero, bottom=[i+j for i,j in zip(m_one, m_two)], color='#FF8787', edgecolor='white', width=barWidth, label = 'Class '+classes[2])
    plt.bar(r, one, bottom=[i+j+k for i,j,k in zip(m_one, m_two,zero)], color='#F8C4B4', edgecolor='white', width=barWidth, label = 'Class '+classes[3])
    plt.bar(r, two, bottom=[i+j+k+l for i,j,k,l in zip(m_one, m_two,zero,one)], color='#E5EBB2', edgecolor='white', width=barWidth,label = 'Class '+classes[4])

    plt.xticks(r, names)
    plt.xlabel("Clients")
    plt.xticks(rotation=90)
    plt.ylabel("Distribution (%)")
    plt.legend()
    plt.title('Class distribution on all FL clients')
    # Show graphic
    plt.show()
    fig.savefig("Results/class_distributions_Synthetic.pdf", dpi = 600,bbox_inches='tight')

plot_barcharts()












