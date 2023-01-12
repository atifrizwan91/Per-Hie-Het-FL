# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 15:41:06 2022

@author: user
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import seaborn as sns


# countries = ['data_0','data_1',"data_2","data_3","data_4", "data_5","data_6","data_7","data_8","data_9"]
# countries = ['syn_data_0','syn_data_1',"syn_data_2","syn_data_3","syn_data_4", "syn_data_5","syn_data_6","syn_data_7","syn_data_8","syn_data_9"]
# labels = ["Client 1","Client 2","Client 3","Client 4","Client 5","Client 6","Client 7","Client 8","Client 9","Client 10"]
all_data = {}


data = 'Thermal'
# Training Acc DNN
metric = 'loss'
model = 'LSTMCNN'
reverse = False
if(data == 'Synthetic'):
    countries = ['syn_data_0','syn_data_1',"syn_data_2","syn_data_3","syn_data_4", "syn_data_5","syn_data_6","syn_data_7","syn_data_8","syn_data_9"]
    labels = ["Client 1","Client 2","Client 3","Client 4","Client 5","Client 6","Client 7","Client 8","Client 9","Client 10"]
    if(model == 'DNN'):
        for i in countries:
            df = pd.read_csv(i+'/performance DNN.csv', names=['unnamed','loss','acc','w_acc','val_loss','val_acc','val_w_acc',])
            all_data[i] = df
        if(metric == 'acc'):
                title = "Training Accuracy"
                x_lab = "Local Epochs"
                y_lab = "Accuracy"
                p_s = [[5,30],[15,30]] # Starting Points
                p_e = [[0.2,0.3],[0.2,0.3]] # Ending Point
                p_msg = [30,0.3]
                m = 'Stuck In local Minima'
                
                p_s1 = [[40,60],[45,60]] # Starting Points
                p_e1 = [[0.65,0.4],[0.65,0.4]] # Ending Point
                p_msg1 = [60,0.4]
                m1 = 'One Server Round'
        if(metric == 'loss'):
                title = "Training Loss"
                x_lab = "Local Epochs"
                y_lab = "Loss"
                p_s = [[5,30],[15,30]] # Starting Points
                p_e = [[1.6,1.3],[1.6,1.3]] # Ending Point
                p_msg = [30,1.3]
                m = 'Stuck In local Minima'
                
                p_s1 = [[40,60],[45,60]] # Starting Points
                p_e1 = [[0.9,1.4],[0.95,1.4]] # Ending Point
                p_msg1 = [60,1.4]
                m1 = 'One Server Round'
        if(metric == 'val_acc'):
                title = "Validation Accuracy"
                x_lab = "Local Epochs"
                y_lab = "Accuracy"
                
        if(metric == 'val_loss'):
                title = "Validation Loss"
                x_lab = "Local Epochs"
                y_lab = "Loss"
                
    if(model == 'LSTMCNN'):
        for i in countries:
            df = pd.read_csv(i+'/performance.csv', names=['unnamed','loss','acc','w_acc','val_loss','val_acc','val_w_acc',])
            all_data[i] = df
        if(metric == 'acc'):
                title = "Training Accuracy"
                x_lab = "Local Epochs"
                y_lab = "Accuracy"
                
        if(metric == 'loss'):
                title = "Training Loss"
                x_lab = "Local Epochs"
                y_lab = "Loss"
                
        if(metric == 'val_acc'):
                reverse = True
                title = "Validation Accuracy"
                x_lab = "Local Epochs"
                y_lab = "Accuracy"
                
        if(metric == 'val_loss'):
                reverse = True
                title = "Validation Loss"
                x_lab = "Local Epochs"
                y_lab = "Loss"

if(data == 'Thermal'):
    labels = ['India', 'Italy', 'Pakistan', 'Philippines', 'Portugal', 'Singapore', 'Sweden', 'Thailand', 'Tunisia','UK']
    countries = ['India', 'Italy', 'Pakistan', 'Philippines', 'Portugal', 'Singapore', 'Sweden', 'Thailand', 'Tunisia','UK']
    if(model == 'DNN'):
        for i in countries:
            df = pd.read_csv(i+'/performance DNN.csv', names=['unnamed','loss','acc','w_acc','val_loss','val_acc','val_w_acc',])
            all_data[i] = df
        if(metric == 'acc'):
                title = "Training Accuracy"
                x_lab = "Local Epochs"
                y_lab = "Accuracy"
               
        if(metric == 'loss'):
                title = "Training Loss"
                x_lab = "Local Epochs"
                y_lab = "Loss"
        if(metric == 'val_acc'):
                title = "Validation Accuracy"
                x_lab = "Local Epochs"
                y_lab = "Accuracy"
                
        if(metric == 'val_loss'):
                title = "Validation Loss"
                x_lab = "Local Epochs"
                y_lab = "Loss"
                
    if(model == 'LSTMCNN'):
        for i in countries:
            df = pd.read_csv(i+'/performance.csv', names=['unnamed','loss','acc','w_acc','val_loss','val_acc','val_w_acc',])
            all_data[i] = df
        if(metric == 'acc'):
                title = "Training Accuracy"
                x_lab = "Local Epochs"
                y_lab = "Accuracy"
                
        if(metric == 'loss'):
                title = "Training Loss"
                x_lab = "Local Epochs"
                y_lab = "Loss"
                
        if(metric == 'val_acc'):
                reverse = False
                title = "Validation Accuracy"
                x_lab = "Local Epochs"
                y_lab = "Accuracy"
                
        if(metric == 'val_loss'):
                reverse = False
                title = "Validation Loss"
                x_lab = "Local Epochs"
                y_lab = "Loss"
def prepare_loss_and_acc(all_results):
    for i in countries:
        pass



def plot_results(all_results):
    fontsize = 16
    fig = plt.figure(figsize=(12,6))
    for i,j in zip(countries,labels):
        df = all_data[i]
        #df = df[:20]
        y = list(df[metric])
        if(reverse):
            y.reverse()
        x_axis = df.index
       
        # add = [0.2]*len(y)
        # y = [i-j for i,j in zip(y,add)]
        linewidth= 1
        if(i == 'Medium_US'):
            linewidth=1
            plt.plot(x_axis,y, label = j, linewidth=linewidth, color = 'red', marker = '*')
            continue
        plt.plot(x_axis,y, label = j, linewidth=linewidth, marker = '*')
    plt.xlabel(x_lab, fontsize=fontsize)
    sns.despine(top=True)
    plt.title(title, fontsize = fontsize+5)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylabel(y_lab, fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    # plt.legend(bbox_to_anchor=(1.01, 1.05),fontsize=12)
    if((metric == 'acc' or metric == 'loss') and model == 'DNN' and data == 'Synthetic'):
        plt.plot(p_s[0],p_e[0], 'r--')
        plt.plot(p_s[1],p_e[1], 'r--')
        plt.text(p_msg[0],p_msg[1], m, fontsize = 12,bbox = dict(facecolor = 'red', alpha = 0.5))
        
        plt.plot(p_s1[0],p_e1[0], 'g--')
        plt.plot(p_s1[1],p_e1[1], 'g--')
        plt.text(p_msg1[0],p_msg1[1], m1, fontsize = 12,bbox = dict(facecolor = 'green', alpha = 0.5))
        
    plt.show()
    fig.savefig('Results/Thermal_'+model+'_'+metric+'.pdf',dpi = 600,bbox_inches='tight')
    
plot_results(all_data)
    
    
    