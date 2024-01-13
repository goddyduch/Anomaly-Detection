"""---------main----------"""
from dataset import *
from modele import * 
from dataset import *

from RNN import *

import scipy
from scipy import stats
from sklearn.base import BaseEstimator

import seaborn as sns
import pandas as pd 
import numpy as np 

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from torch import torch

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE




dataset1 = dataset("Ferrari","data_cpu.csv")
dataset1.importation()
dataset1.information()
dataset1.visualisation()

dataset1.feature_engeenering()
#dataset1.visualisation_tendance("day")
#dataset1.visualisation_tendance("hour")


#-----------------------Modele de clustering - ou modèle spécialement détection d'anomalie --------------------
ISF = MyEstimator(IsolationForest(contamination=0.03))
LOF = MyEstimator(LocalOutlierFactor(n_neighbors=8,contamination=0.03))
epsilon = 4
minpoint = 2* dataset1.df.shape[1]

DBSCAN = MyEstimator(DBSCAN(eps = epsilon, min_samples = 4))
GMM = MyEstimator(GaussianMixture(n_components=7, n_init=5, random_state=42))
Autoencoder = MyEstimator(AutoEncoder(hidden_neurons =[10, 10, 2, 10, 10],epochs=75,contamination=0.03))


def testing(model): 
    model.fit(dataset1.df)
    model.predict()
    model.evaluation()
    return str(model)


print(testing(DBSCAN))
print(testing(LOF))
print(testing(ISF))
print(testing(GMM)) 
print(testing(Autoencoder)) 


"""----------RNN--------------"""
"""
df_test = dataset1.df[["value","hour","day","min","daylight","weekday","avgrolling","workday", "max"]]
value =list(df_test["value"])
datasize = len(list(df_test["value"]))
unroll_length = 15
prediction_time = 1 
testdatasize = 500

testdatacut = testdatasize   + unroll_length  

x = unroll(df_test,unroll_length)
x_train, y_train, x_test, y_test = split(x,value,testdatasize,testdatacut,unroll_length)
n_in = int(x_train.size(dim=1))
n_train = int(x_train.size(dim=0))
n_test = int(y_test.size(dim=0))
entrainement = False


if entrainement : 
    model = LSTM(n_in)
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1, momentum=0.9)
    model, loss, hat_y = fit_model(model,criterion,optimizer,500, x_train, y_train)
    torch.save(model.state_dict(), 'model.pt')
    
else : 
    model = torch.load('model.pt')
    model = LSTM(n_in)
    model.load_state_dict(torch.load('model.pt'))


hat_y = model(x_train)
y_test, prediction , y_train, hat_y = validation(hat_y,x_train, y_train, x_test, y_test,model,n_train, n_test)
anomalie = anomalie(y_test, prediction, 0.03,n_test)

"""
