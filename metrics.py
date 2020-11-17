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

%matplotlib inline

def metrics(y_true,y_pred,y_probs):
    roc = roc_auc_score(y_true,y_probs)
    print("The ROC AUC Score is: {}".format(roc))
    cf = confusion_matrix(y_true,y_preds)
    print("The Confusion Matrix is:")
    print(cf)
    f1score = f1_score(y_true, y_preds)
    print("The F1 score is: {}".format(f1score))
    fpr,tpr,_ = roc_curve(y_true,y_preds)
    print("ROC_AUC curve")
    pyplot.plot(fpr, tpr, marker='.', label='ROC_AUC')


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
  print("Confusion Matrix for all:")
  print(cf_total)

  
  x = list(range(1,5))  
  plt.plot(x,roc_l,label='ROC')
  plt.plot(x,f1_score_l,label='F1_score')
  plt.ylabel('Split Vs ROC')
  plt.show()