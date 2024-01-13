"""---------Importation module----------"""
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

from pyod.models.vae import VAE


"""-----------------VAE ------------------------"""
dataset1 = dataset("Ferrari","data_cpu.csv")
dataset1.importation()
dataset1.feature_engeenering()
df = dataset1.df[["value","hour","day","min","daylight","weekday","avgrolling","workday", "max"]]

vae = VAE(encoder_neurons=[8,4,1],decoder_neurons=[8,4,1],epochs=150, contamination=0.03)

vae.fit(df)
anomalie = vae.predict(df)
print("anomalie",anomalie)

test = pd.DataFrame(list(zip(df["value"], anomalie)),columns =['value', 'anomalie'])
titre = "VAE" + " avec " + "3%" + " d'anomalies " 


#affichage anomalie 
fig, axes = plt.subplots(2, 1)
fig.suptitle(titre, fontsize=16)
sns.lineplot(data=test, x=test.index, y="value", color="blue",ax= axes[0])
sns.scatterplot(data = test, x=test.index, y="value" ,hue="anomalie",palette=['b', 'r'],markers=True, ax= axes[0])
sns.histplot(data=test, x="value", bins=20, hue="anomalie", palette=['b', 'r'],ax=axes[1])
axes[1].legend(labels=["Non anomalie","Anomalie"])
chemin = "Result/" + "VAE" + ".png"
plt.savefig(chemin)
plt.show()

#affichage loss 
#loss1 = pd.DataFrame(vae.history_.get('loss'), columns=["loss"])
#val_loss1 = pd.DataFrame(vae.history_.get('val_loss'),columns=["val_loss"])
#fig, axes =plt.subplots(2,1)
#sns.lineplot(data=loss1, x=loss1.index, y="loss",ax=axes[0])
#sns.lineplot(data=val_loss1, x=loss1.index, y="val_loss",ax=axes[0])
#plt.show()