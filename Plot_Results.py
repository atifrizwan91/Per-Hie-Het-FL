
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

# countries = ['India', 'Italy', 'Pakistan', 'Philippines', 'Portugal', 'Singapore', 'Sweden', 'Thailand', 'Tunisia','UK' ]
countries = ['data_0','data_1',"data_2","data_3","data_4", "data_5","data_6","data_7","data_8","data_9"]
def plot_metric(losses,countries, ptype):
    x = [x for x in range(0, len(losses[0]))]
    fig = plt.figure()
    for loss, c in zip(losses,countries):
        plt.plot(x,loss, label = c)
    plt.legend()
    plt.xticks([i for i in range(0, len(x),5)])
    plt.show()
    fig.savefig(ptype+'.pdf', dpi=600)

def plot_loss_and_acc(country):
    acc = pd.read_csv('Acc_all_countries.csv')[country].values.tolist()
    loss = pd.read_csv('losses_all_countries.csv')[country].values.tolist()
    x = [x for x in range(0, len(loss))]
    fig = plt.figure()
    plt.plot(x,loss, color = '#123')
    plt.plot(x,acc, color = '#A31')
    plt.title('Philippines')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy and Loss')
    plt.show()
    #fig.savefig(country+'.png', dpi=600)
    
    
def read_acc_loss__data(folders):
    Acc = []
    losses = []
    temp = []
    for i in folders:
        df = pd.read_csv(i+'/loss.csv')
        temp.append(len(list(df.iloc[:, 0])))
    min_t = min(temp)
    print(min_t)
    for i in folders:
        df = pd.read_csv(i+'/loss.csv', nrows=min_t)
        losses.append(df.iloc[:, 0].values.tolist())
        Acc.append(df.iloc[:, 1].values.tolist())
    return losses,Acc

def mergeHistory(folders):
    merged = pd.DataFrame()
    for i in folders:
        df = pd.read_csv(i+'/performance.csv')
        merged = pd.concat([merged, df], axis = 1)
    return merged


def mergee_waiting_training_time(countries,time_type):
    w_time = {}
    temp = []
    for i in countries:
        df = pd.read_csv(i+'/'+ time_type +'.csv')
        temp.append(len(list(df.columns)))
    min_t = min(temp)
    for i in countries:
        df = pd.read_csv(i+'/'+ time_type +'.csv')
        print()
        w_time[i] = list(df.columns)[:min_t]
    df = pd.DataFrame(w_time)
    df.to_csv(time_type+'.csv')



#mergee_waiting_training_time(countries,'training_time') #training_time   #waiting_time



# losses,acc = read_acc_loss__data(countries)

# df_losses  = pd.DataFrame(np.array(losses).T, columns=countries)
# df_acc  = pd.DataFrame(np.array(acc).T, columns=countries)
# df_acc.to_csv('Acc_all_countries.csv')
# df_losses.to_csv('Losses_all_countries.csv')


history = mergeHistory(countries)
history.to_csv('CombinedHistory.csv')

for i in countries:
    plot_loss_and_acc(i)
# plot_metric(losses,countries, 'Loss')
# plot_metric(acc,countries,'Accuracy')