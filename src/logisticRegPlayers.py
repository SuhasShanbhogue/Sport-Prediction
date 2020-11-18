import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
 
 
def metrics(y_true,y_preds,y_probs):
    roc = roc_auc_score(y_true,y_probs)
    print("The ROC AUC Score is: {}".format(roc))
    f1score = f1_score(y_true, y_preds)
    print("The F1 score is: {}".format(f1score))
    cf = confusion_matrix(y_true,y_preds)
    print("The Confusion Matrix is:")
    # plot_confusion_matrix(clf, X_test, y_test,
    #                       cmap=plt.cm.binary)
    cm =confusion_matrix(y_true, y_preds)  
    index = [1, 0]  
    columns = [1, 0]  
    cm_df = pd.DataFrame(cm,columns,index)                      
    plt.figure(figsize=(10,6))  
    sns.heatmap(cm_df, annot=True, cmap=plt.cm.binary)
    print("----------------------------------------\n ")
    #plt.show()
 
def metrics_combined(comb_y_true,comb_y_pred,comb_y_probs):
    roc =0
    f1score =0
    roc_l=[]
    f1_score_l=[]
    cf_total=0
    for i in range(4):
        roc += roc_auc_score(comb_y_true[i],comb_y_probs[i])
        roc_l.append(roc_auc_score(comb_y_true[i],comb_y_probs[i]))
        f1score += f1_score(comb_y_true[i],comb_y_pred[i])
        f1_score_l.append(f1_score(comb_y_true[i],comb_y_pred[i]))
        cf = confusion_matrix(comb_y_true[i],comb_y_pred[i])
        cf_total += cf

    print("Mean ROC score:{}".format(roc/4))
    print("Mean F1 score{}".format(f1score/4))
    # cm =confusion_matrix(y_test, y_preds)  
    index = [1, 0]  
    columns = [1, 0]  
    cm_df = pd.DataFrame(cf_total,columns,index)                      
    plt.figure(figsize=(10,6))  
    sns.set(font_scale=1.2)
    plot = sns.heatmap(cm_df, annot=True, cmap=plt.cm.binary)
    plt.show()
    fig = plot.get_figure()
    fig.savefig("../content/cf for LR players.png")
    x = ["2012", "2013", "2014", "2015"] 
    plt.plot(x,roc_l,label='ROC', linestyle = '--', color = 'black')
    plt.plot(x,f1_score_l,label='F1_score', linestyle = '-', color = 'gray')
    plt.legend()
    plt.ylabel('ROC/F1-score')
    plt.savefig("../content/Split VS ROC-F1-score for LR Players.png")
    plt.show()

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
comb_y_true=[]
comb_y_pred=[]
comb_y_probs=[]

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
    print("AFTER TUNING we get :")   
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
        y_probs = LR.predict_proba(X_test)
        comb_y_true.append(y_test)
        y_probs = y_probs[:,1]
        comb_y_probs.append(y_probs)
        comb_y_pred.append(y_preds)
        confusion_matrix(y_test,y_preds)
        roc = roc_auc_score(y_test,y_preds)
        total += roc
        print("{} ROC is {}".format(2012+i,roc))
        metrics(y_test,y_preds,y_probs)
    metrics_combined(comb_y_true,comb_y_pred,comb_y_probs)

main()