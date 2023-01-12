# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 00:48:18 2022

@author: user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# countries = ['syn_data_0','syn_data_1',"syn_data_2","syn_data_3","syn_data_4", "syn_data_5","syn_data_6","syn_data_7","syn_data_8","syn_data_9"]
countries = ['India', 'Italy', 'Pakistan', 'Philippines', 'Portugal', 'Singapore', 'Sweden', 'Thailand', 'Tunisia','UK']
metrics = ['loss','val_loss','acc','val_acc']
titles = ['Training Loss','Validation Loss','Training Accuracy','Validation Accuracy']
t = 0
fig, axs = plt.subplots(2,2)
r,c = 0,0
for metric in metrics:
    all_data = []
    for i in countries:
        df = pd.read_csv(i+'/performance.csv', names=['unnamed','loss','acc','w_acc','val_loss','val_acc','val_w_acc',])
        df = df[df['unnamed'] == 4]
        df = df[metric]
        if(metric == 'val_acc' or metric == 'val_loss'):
            l = df.values.tolist()
            l.reverse()
            all_data.append(l)
        else:
            all_data.append(df.values.tolist())
        # all_data.append(df.values.tolist())
    
    all_data = np.array(all_data)
    avg_p = np.average(all_data, axis=0)
    fontsize = 16
    
    for i in all_data:
        x_axis = [j for j in range(0, len(i))]
        axs[r,c].plot(x_axis, i, marker = '*')
    axs[r,c].plot(x_axis, avg_p, color = 'black', marker = 'o', label = 'Global')
    axs[r,c].legend()
    axs[r,c].set_title(titles[t])
    c +=1
    
    t += 1
    if c == 2:
        c = 0
        r+= 1
fig.tight_layout()
fig.savefig('Results/Global Loss and acc Synthetic LSTM.pdf',dpi = 600,bbox_inches='tight')