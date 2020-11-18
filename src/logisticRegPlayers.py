import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

d_2011 = pd.read_csv('../Data/Players_final_2011.csv')
d_2012 = pd.read_csv('../Data/Players_final_2012.csv')
d_2013 = pd.read_csv('../Data/Players_final_2013.csv')
d_2014 = pd.read_csv('../Data/Players_final_2014.csv')
d_2015 = pd.read_csv('../Data/Players_final_2015.csv')

tunedC = 1

def tuning():
    list_years = [d_2011,d_2012,d_2013,d_2014,d_2015]

    print("Logistic Regression Tuning for player Data: ")
    print("Time Series Split  :")


    C = np.logspace(-100, 10, 200)
    maxroc = 0
    minvar = 100000
    tunedC = 1
    for c in C:
        total =[]
        for i in range(4):
            frames=[]
            for j in range(i+1):
                frames.append(list_years[j])
                train = pd.concat(frames)
            test = list_years[i+1]
            X_train = train.drop(['Winner'],axis=1)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            y_train = train['Winner']
            X_test = test.drop(['Winner'],axis=1)
            X_test = scaler.transform(X_test)
            y_test = test['Winner'] 
            LR = LogisticRegression(C=c,random_state=6,solver='liblinear',max_iter=100000).fit(X_train, y_train)
            y_preds = LR.predict(X_test)
            confusion_matrix(y_test,y_preds)
            roc = roc_auc_score(y_test,y_preds)
            total.append(roc)
          
        if(np.var(total)<minvar and np.mean(total)>0.5):
            minvar = np.var(total)
            print(minvar)
            tunedC = c
        # if(np.mean(total))>maxStat:
        #     maxStat = np.mean(total)
        #     print(maxStat)
        #     tunedC = c
    print("FINAL Results :")   
    print("C : " + str(tunedC))
    #print("RUCAUC : "+ str(maxStat))


def main():
    

    print("{} shape: {}".format(2011,d_2011.shape))
    print("{} shape: {}".format(2012,d_2012.shape))
    print("{} shape: {}".format(2013,d_2013.shape))
    print("{} shape: {}".format(2014,d_2014.shape))
    print("{} shape: {}".format(2014,d_2015.shape))
    tuning()

    list_years = [d_2011,d_2012,d_2013,d_2014,d_2015]

    print("Logistic Regression: ")

    total =0
    for i in range(4):
        frames=[]
        for j in range(i+1):
            frames.append(list_years[j])
        train = pd.concat(frames)
        test = list_years[i+1]
        X_train = train.drop(['Winner'],axis=1)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        y_train = train['Winner']
        X_test = test.drop(['Winner'],axis=1)
        X_test = scaler.transform(X_test)
        y_test = test['Winner']
        LR = LogisticRegression(C=tunedC,random_state=6,max_iter=100000,solver='liblinear').fit(X_train, y_train)
        y_preds = LR.predict(X_test)
        confusion_matrix(y_test,y_preds)
        roc = roc_auc_score(y_test,y_preds)
        total += roc
        print("{} ROC is {}".format(2012+i,roc))
    
    print(total/4)


main()