from dataset import *

from sklearn.base import BaseEstimator
import seaborn as sns
import numpy as np 


class MyEstimator(BaseEstimator):
    
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.prediction = None
        self.score = None
        self.param = None
        self.anomalie = None
        self.X = None
        self.name = None
        self.loss =None
        

    def fit(self, X):
        self.X = X
        print(str(self.base_estimator)[0:6])
        if  str(self.base_estimator)[0:6] ==  "LocalO" or str(self.base_estimator)[0:6] == "Gaussi": 
            self.prediction = self.base_estimator.fit_predict(X)

        elif str(self.base_estimator)[0:4] == "Auto" : 
            data_AE = self.X[["value",'avgrolling']]
            self.base_estimator.fit(data_AE)
        else : 
            self.base_estimator.fit(X)

        return self.base_estimator


    def predict(self):
        if  str(self.base_estimator)[0:6] ==  "LocalO" :
            self.name = "Local outlier factor"
            self.score = self.base_estimator.negative_outlier_factor_

            
        elif str(self.base_estimator)[0:6] ==  "DBSCAN" :
            self.name = "DBSCAN"
            label = self.base_estimator.labels_
            label= np.where(label != -1, 1,-1)
            self.prediction = label
            
            
        elif str(self.base_estimator)[0:6] == "Gaussi" :
            self.name = "Gaussian Mixture modele"
            self.score = self.base_estimator.score_samples(self.X)
            pct_threshold = np.percentile(self.score, 3)
            pred = pd.DataFrame()
            pred["score"] = self.score 
            self.prediction= pred["score"].apply(lambda x: 1 if x > pct_threshold else -1)

        elif str(self.base_estimator)[0:4] == "Auto" : 
            self.name = "AutoEncoder"
            data_AE = self.X[["value",'avgrolling']]
            self.prediction = self.base_estimator.predict(data_AE)
            self.prediction = np.where(self.prediction==1, -1, 1)
            
            #choix nombre d'époch
            """self.loss = pd.DataFrame(self.base_estimator.history_.get('loss'), columns=["loss"])
            self.loss = pd.DataFrame(self.base_estimator.history_.get('loss'), columns=["loss"])
            val_loss1 = pd.DataFrame(self.base_estimator.history_.get('val_loss'),columns=["val_loss"])


            fig, axes =plt.subplots(2,1)
            sns.lineplot(data=self.loss, x=self.loss.index, y="loss",ax=axes[0])
            sns.lineplot(data=val_loss1, x=val_loss11.index, y="val_loss",ax=axes[0])
            plt.show()"""  
            

        else : 
            self.prediction = self.base_estimator.predict(self.X)
            self.score = self.base_estimator.score_samples(self.X)
            self.name = "Isolation forest"
       
        self.param = self.base_estimator.get_params(deep=False)
        return 
    

    def evaluation(self) :
        print(self.prediction)
        anomalieP = pd.DataFrame(self.prediction).value_counts(normalize=True).mul(100).round(1).astype(str)
        titre = self.name + " avec " + str(anomalieP[-1]) + " % Anomalie " 
        
        resultat = pd.DataFrame(list(zip(self.X["value"], self.prediction)),columns =['value', 'anomalie'])
        fig, axes = plt.subplots(2, 1)
        fig.suptitle(titre, fontsize=16)
        
        sns.lineplot(data = resultat, x=resultat.index, y = "value",color="blue",ax=axes[0])
        sns.scatterplot(data= resultat, x=resultat.index, y = "value", hue= "anomalie", palette=["r", "b"],markers=True,ax=axes[0])
        sns.histplot(data=resultat, x="value", bins=20, hue="anomalie", palette=['lightgreen', 'pink'],ax=axes[1])
        axes[1].legend(labels=["Non anomalie","Anomalie"])
       
        
        print("hello3")
        chemin = "Result/" + str(self.base_estimator)[:10] + ".png"

        plt.savefig(chemin)
        plt.show()
       
       










 
 