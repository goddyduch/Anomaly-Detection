from dataset import *
from modele import * 

from torch import torch
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import numpy     
import matplotlib.pylab as plt
import numpy as np
import torch.nn as nn



def unroll(data,length):
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    data = pd.DataFrame(data)
    previous = []
    for index in range(length,data.shape[0]) :
        previou = []
        previou = list(data.iloc[(index- length):(index-1),0])
        previou.append(data.iloc[index,1])
        previou.append(data.iloc[index,2])
        previou.append(data.iloc[index,3])
        previou.append(data.iloc[index,4])
        previou.append(data.iloc[index,5])
        previou.append(data.iloc[index,6])
        previou.append(data.iloc[index,7])
        previou.append(data.iloc[index,8])
        
        
        previous.append(previou)
    previous = pd.DataFrame(previous)
    return previous



def split(x,data_n,testdatasize,cut,length) :
    x_train = np.array(x[0:-cut-1])
    y_train = np.array(data_n[length+1:-cut])

    # test data
    x_test = np.array(x[0-cut:-1])
    y_test = np.array(data_n[1-cut:])
    

    # see the shape
    x_train = (torch.tensor(np.array(x_train))).float()
    y_train = (torch.tensor(np.array(y_train))).float()
    x_test = (torch.tensor(np.array(x_test))).float()
    y_test = (torch.tensor(np.array(y_test))).float()

    return x_train, y_train, x_test, y_test

    


#"""architecture : LSTM - LSTM - DENSE LAYER - MSE LOSS -"""


class LSTM(nn.Module):
    def __init__(self, n_in):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(n_in, 100)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(100,50 )
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(50,25 )
        self.dropout3 = nn.Dropout(0.2)
        self.linear = nn.Linear(25,1)
        

    def forward(self, x):
        out1, states = self.lstm(x)
        out2 = self.dropout1(out1)
        out3, states = self.lstm2(out2)
        out4 = self.dropout2(out3)
        out5, states = self.lstm3(out4)
        out6 = self.dropout3(out5)
        out7 = self.linear(out6)
        return out7


def fit_model(model,criterion,optimizer,nbr_epoch,x_train, y_train):
    loss_l = [] 
    for num_epoch in range(nbr_epoch): 
            optimizer.zero_grad()
            hat_y = model(x_train)
            loss = criterion(hat_y, y_train)
            loss.backward()
            optimizer.step() 
            loss_l.append(loss.item())
            if num_epoch % 5 == 0:
                print('epoch {}, loss {}'.format(num_epoch, loss.item()))
    return model , loss_l, hat_y
            

def validation(hat_y,x_train, y_train, x_test, y_test,model,n_train, n_test): 
    hat_y= hat_y.detach().numpy()
 
    hat_y = list(np.reshape(hat_y, n_train)) 
    prediction = model(x_test)
    prediction = prediction.detach().numpy()
    prediction = list(np.reshape(prediction,n_test))
    
    #visualisation 
    fig, axes = plt.subplots(2, 1)
    sns.color_palette("flare", as_cmap=True)
    axes[0].plot(y_train,color="blue",label = 'Données')
    axes[0].plot(hat_y ,color="red", label = 'Prédiction')
    axes[0].legend()
    axes[0].set_title("Entrainement")

    axes[1].plot(y_test,color="blue",label = 'Données')
    axes[1].plot(prediction ,color="red",label = 'Prédiction')
    axes[1].legend()
    axes[1].set_title("Prédiction")
    plt.savefig('Result/Train_RNN.png')
    plt.show()
   

    return y_test, prediction , y_train, hat_y


def anomalie(y_test, prediction, peroutlier,prediction_size):
    difference = np.array([abs(y_test - prediction) for y_test , prediction in zip(y_test, prediction)])

    number_of_outliers = int(peroutlier*len(difference))
    threshold = int(pd.DataFrame(difference).nlargest(number_of_outliers,0).min())
    test = [int(x>=threshold) for x in difference]

    #visualisation 
    titre = "RNN  avec "+  str(peroutlier*100) +" pourcents d'anomalies"
    fig, axes = plt.subplots(2, 1)
    fig.suptitle(titre, fontsize=16)
    sns.lineplot(x=range(prediction_size), y =y_test,color="blue",ax=axes[0])
    sns.scatterplot(x=range(prediction_size) , y = y_test, hue=test, palette=[ "b","r",], markers=True,ax=axes[0])
    sns.histplot(data=np.array(range(prediction_size)), x= y_test, bins=20, hue=test,palette=['lightgreen', 'pink'], ax=axes[1])
    axes[1].legend(labels=["Non anomalie","Anomalie"])
    plt.savefig('Result/RNN_anomalie.png')
    plt.show()
    
   
    
    return "done"
        
