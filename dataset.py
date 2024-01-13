import scipy
from scipy import stats
import matplotlib

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 

from sklearn.preprocessing import LabelEncoder



class dataset : 
    def __init__(self, nom, path, outliers_fraction=0.01):
        self.nom = nom
        self.path = path
        self.df = None
        self.col = 0
        self.raw = 0
        
    def setdf_modif(self, df_modif) : 
        self.df_modif = df_modif
        
    def setdf(self, df):
        self.df = df
    
    def setraw(self, df):
        self.raw = df.shape[0]
        
    def setcol(self, df):
        self.col = df.shape[1]
        
        

    def importation(self):
        df = pd.read_csv(self.path)
        dataset.setdf(self,df)
        dataset.setcol(self, df)
        dataset.setraw(self,df)


    def information(self):
        print("DATASET", self.df.head(5))
        print("")
        print("Info", self.df.info())
        print("DESCRIPTION",self.df["value"].describe())
        print(" ------------     ")
        print("Nbr colonne",self.col)
        print("Nbr raw",self.raw)
    
        
  
        date = pd.to_datetime(self.df["timestamp"])
        print("Unique year", date.dt.year.unique())
        print("Unique hour", date.dt.hour.unique())
        print("Unique minute", date.dt.minute.unique())
        print("Debut", date[0])
        print("Fin", date[self.raw-1])

    def visualisation(self):
        fig, ax = plt.subplots(2)
        index = [x for x in range(self.raw)]
        ax[0].plot(index,self.df["value"])
        ax[0].set_xlabel("temps")
        ax[0].set_ylabel("CPU")
        ax[0].set_title("Consommation du CPU en fonction du temps")
        
        ax[1].set_xlabel("cpu")
        ax[1].set_ylabel("Quantités")
        ax[1].hist(self.df["value"], bins=20,edgecolor = 'red')
        ax[1].set_title("Histogramme de fréquence ")
        plt.show()
        
       

    def visualisation_tendance(self, temps):
        df_time = pd.DataFrame()
        df_time["value"] = self.df["value"]#[:200]
        df_time[temps] = self.df[temps]
        df_time_max  = df_time.groupby(temps).agg({"value": "max"})
        df_time_mean  = df_time.groupby(temps).agg({"value": "mean"})
        
       
        fig, ax = plt.subplots(2)
        ax[0].set_xlabel(temps)
        ax[0].set_ylabel("Temperature moyenne")
        ax[0].bar(df_time_mean.index, df_time_mean["value"], label="mean")
        ax[0].set_title("Consommation moyenne de CPU par  " + temps)
        
    
        ax[1].set_xlabel(temps)
        ax[1].set_ylabel(" CPU maximum ")
        ax[1].bar(df_time_max.index, df_time_max["value"], label="max")
        ax[1].set_title("Consommation maximum de CPU par " + temps )
        plt.show()
        


        

    def feature_engeenering(self) : 
        df_modif = self.df.copy()
    
        """ ------------     Extraction year, month, hour  -----------------------------------------"""
        df_modif["timestamp"] = pd.to_datetime(df_modif["timestamp"]) #format="%Y-%m-%d %H:%M:%S.%f")
        print(df_modif["timestamp"])
        df_modif["year"] = df_modif["timestamp"].dt.year 
        df_modif["month"] = df_modif["timestamp"].dt.month
        df_modif["day"] = df_modif["timestamp"].dt.day
        df_modif["hour"] = df_modif["timestamp"].dt.hour
        df_modif["min"] = df_modif["timestamp"].dt.minute
        df_modif["weekday"] = df_modif["timestamp"].dt.day_name()
        
        
        #ENCODAGE
        enc = LabelEncoder()
        x = list(set(df_modif["weekday"]))
        enc.fit(x)
        df_modif["weekday"] = enc.transform(df_modif["weekday"])
      
        
        #AJOUT VARIABLE INTESSANTE
        df_modif["daylight"] =  ((df_modif["hour"]>8) & (df_modif["hour"]<18)).astype(int)
        df_modif['workday'] = (df_modif['weekday'] < 5).astype(int)
        
        
        #ROLLING 
        df_modif["avgrolling"] = df_modif.value.rolling(20).mean()
        df_modif["max"] = df_modif.value.rolling(10).mean()
        df_modif["before"] = df_modif.value.rolling(1).mean()
        

        """"---------------SETTER------------------- """
        df_modif = df_modif.iloc[20:]
        df_modif = df_modif.drop(columns =["timestamp"])
        self.df = df_modif
        dataset.setcol(self, self.df)
        dataset.setraw(self,self.df)
        
        print(self.df)
        
        