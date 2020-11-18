import tensorflow 
import keras.layers.advanced_activations
from numpy.random import seed
seed(1)
from sklearn.preprocessing import StandardScaler
tensorflow.random.set_seed(2)
from sklearn.metrics import confusion_matrix
import pandas as pd
from tensorflow.keras import regularizers
from keras.layers.core import Dropout
from tensorflow.keras import layers
from tensorflow.keras import initializers
from sklearn.metrics import roc_auc_score
from numpy.random import seed
seed(1)
import seaborn as sns

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
    print("----------------------------- ")
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
    plot =sns.heatmap(cm_df, annot=True, cmap=plt.cm.binary)
    plt.show()
    fig = plot.get_figure()
    fig.savefig("../content/cf for ANN players.png")
    x = ["2012", "2013", "2014", "2015"] 
    plt.plot(x,roc_l,label='ROC', linestyle = '--', color = 'black')
    plt.plot(x,f1_score_l,label='F1_score', linestyle = '-', color = 'gray')
    plt.legend()
    plt.ylabel('ROC/F1-score')
    plt.savefig("../content/Split VS ROC-F1-score for ANNplayers.png")
    plt.show()

d_2011 = pd.read_csv('../Data/Players_final_2011.csv')
d_2012 = pd.read_csv('../Data/Players_final_2012.csv')
d_2013 = pd.read_csv('../Data/Players_final_2013.csv')
d_2014 = pd.read_csv('../Data/Players_final_2014.csv')
d_2015 = pd.read_csv('../Data/Players_final_2015.csv')

print("{} shape: {}".format(2011,d_2011.shape))
print("{} shape: {}".format(2012,d_2012.shape))
print("{} shape: {}".format(2013,d_2013.shape))
print("{} shape: {}".format(2014,d_2014.shape))
print("{} shape: {}".format(2014,d_2015.shape))
def main():
	model_nn = keras.Sequential([
	  keras.layers.Dense(units=20,activation='relu',kernel_regularizer = regularizers.l1_l2(l1=0.0001,l2=0.0001)),
	  Dropout(0.25),
	  keras.layers.Dense(units=25,activation='relu',kernel_regularizer = regularizers.l1_l2(l1=0.0001,l2=0.0001)),
	  Dropout(0.25),
	  keras.layers.Dense(units=30,activation='relu',kernel_regularizer = regularizers.l1_l2(l1=0.0001,l2=0.0001)),
	  Dropout(0.25),
	  keras.layers.Dense(units=1,activation='sigmoid')
	])

	list_years = [d_2011,d_2012,d_2013,d_2014,d_2015]
	roc_l = []
	print("ANN Neural Network:")
	tunedParameters = {}
	# parameters={'batch_size':[4,8],
	#            'epochs':[50,100],
	#            'optimizer':['adam']}


	# maxScore = 0
	# for batchSize in parameters['batch_size']:
	#   for epoch in parameters['epochs']:
	#     for optim in parameters['optimizer']:
	#       keras.backend.clear_session()
	#       model_nn.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
	#       total =0
	#       for i in range(4):
	#           frames=[]
	#           for j in range(i+1):
	#               frames.append(list_years[j])
	#           train = pd.concat(frames)
	#           test = list_years[i+1]
	#           X_train = train.drop(['Winner'],axis=1)
	#           scaler = StandardScaler()
	#           scaler.fit(X_train)
	#           X_train = scaler.transform(X_train)
	#           y_train = train['Winner']
	#           X_test = list_years[i+1].drop(['Winner'], axis = 1)
	#           X_test = scaler.transform(X_test)
	#           y_test = list_years[i+1]['Winner']
	        
	#           history = model_nn.fit(X_train,y_train,epochs=epoch,batch_size= batchSize,shuffle = False)
	#           y_preds = model_nn.predict(X_test)
	#           for i in range(len(y_preds)):
	#               if y_preds[i]>=0.5:
	#                   y_preds[i]=1
	#               else:
	#                   y_preds[i]=0
	#           confusion_matrix(y_test,y_preds)
	#           roc = roc_auc_score(y_test,y_preds)
	#           roc_l.append(roc)
	#           total += roc

	#       # for i in range(4):
	#       #   print("{} ROC is {}".format(2012+i,roc_l[i]))
	#       print("Average: {}".format(total/4))
	#       if total/4 > maxScore:
	#         maxScore = total/4
	#         print(maxScore)
	#         tunedParameters['batch_size'] = batchSize
	#         tunedParameters['epochs'] = epoch
	#         tunedParameters['optimizer'] = optim

	tunedParameters['batch_size'] = 8
	tunedParameters['epochs'] = 10
	tunedParameters['optimizer'] = 'adam'

	comb_y_true=[]
	comb_y_preds=[]
	comb_y_probs=[]


	list_years = [d_2011,d_2012,d_2013,d_2014,d_2015]
	roc_l = []
	cf_l = []
	print("ANN Neural Network:")

	keras.backend.clear_session()
	model_nn.compile(loss='binary_crossentropy', optimizer=tunedParameters['optimizer'], metrics=['accuracy'])


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
	  X_test = list_years[i+1].drop(['Winner'], axis = 1)
	  X_test = scaler.transform(X_test)
	  y_test = list_years[i+1]['Winner']
	  
	  history = model_nn.fit(X_train,y_train,epochs=tunedParameters['epochs'],batch_size = tunedParameters['batch_size'] ,shuffle=False)
	  y_probs = model_nn.predict(X_test)
	  #y_preds = model_nn.predict(X_train)
	  y_preds =[]
	  for i in range(len(y_probs)):
	      if y_probs[i]>=0.5:
	          y_preds.append(1)
	      else:
	          y_preds.append(0)
	  # cf_l.append(confusion_matrix(y_test,y_preds))
	  # roc = roc_auc_score(y_test,y_preds)
	  # roc_l.append(roc)
	  # total += roc
	  comb_y_true.append(y_test)
	  comb_y_probs.append(y_probs)
	  comb_y_preds.append(y_preds)
	  metrics(y_test,y_preds,y_probs)
	  print("-----------------------------------------\n")
	metrics_combined(comb_y_true,comb_y_preds,comb_y_probs) 

if __name__ == "__main__":
  main()