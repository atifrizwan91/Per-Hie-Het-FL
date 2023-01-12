

import pandas as pd
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
countries = ['syn_data_0','syn_data_1',"syn_data_2","syn_data_3","syn_data_4", "syn_data_5","syn_data_6","syn_data_7","syn_data_8","syn_data_9"]
# countries = ['India', 'Italy', 'Pakistan', 'Philippines', 'Portugal', 'Singapore', 'Sweden', 'Thailand', 'Tunisia','UK']
# df_individual = pd.read_csv('SotaModels\\Synthetic Data Results\\precision_DNN.csv')
df_individual = pd.read_csv('SotaModels\\countries results\\precision_LSTMCNN.csv')
df_individual = df_individual[['TsP1','TsP2','TsP3','TsP4','TsP5']]
r = 0
fig, axs = plt.subplots(10,20,figsize=(20, 20))
for i in countries:
    df = pd.read_csv(i+'/performance_after_round DNN.csv', names=['unnamed','TrainLoss','TrainAcc','TrainPrecision','TestLoss','TestAcc','TestPrecision','TrP1','TrP2','TrP3','TrP4','TrP5','TsP1','TsP2','TsP3','TsP4','TsP5'])
    df_ = df.drop(['unnamed','TrainLoss','TrainAcc','TrainPrecision','TestLoss','TestAcc','TestPrecision','TrP1','TrP2','TrP3','TrP4','TrP5'], axis = 1)
    l = df_.values.tolist()
    cols = [x for x in range(len(l))]
    
    
    j = 0
    print(i)
    # fig.suptitle("Classwise precision of all clients for each round", fontsize=25)
    fig.tight_layout()
    for y in l:
        x = [-2,-1,0,1,2]
        axs[r,j].bar(x, y, width=0.8, edgecolor="white", linewidth=1.2,color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        axs[r,j].get_xaxis().set_visible(False)
        axs[r,j].yaxis.set_tick_params(labelsize=10)
        # axs[r,j].set(xlim=(-2, 2), xticks=np.arange(-2, 3))
        axs[r,j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        j += 1
        if(j == 19):
            break
    axs[r,j].bar(x, df_individual.iloc[[r]].values.tolist()[0], width=0.8, edgecolor="white", linewidth=1.2,color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    axs[r,j].get_xaxis().set_visible(False)
    r += 1

line = plt.Line2D([0.95,0.95],[1,0], transform=fig.transFigure, color="red")
fig.add_artist(line)

fig.savefig('Results/Classwise precision test DNN.pdf',dpi = 600,bbox_inches='tight')