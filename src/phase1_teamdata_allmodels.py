
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


import tensorflow as tf
from tensorflow import keras

from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
# non linear activation
# sigmoid activation

# %matplotlib inline




def metrics(y_true,y_preds,y_probs):
		roc = roc_auc_score(y_true,y_probs)
		print("The ROC AUC Score is: {}".format(roc))
		f1score = f1_score(y_true, y_preds)
		print("The F1 score is: {}".format(f1score))
		cf = confusion_matrix(y_true,y_preds)
		print("The Confusion Matrix is:")
		print(cf)

def confusion_matrix_plot(array):
		 df_cm = pd.DataFrame(array, index = [i for i in ["Loss","Win"]],
							columns = [i for i in ["Predicted_Loss","Predicted_Win"]])
		 sn.set(font_scale=1) # for label size
		 plt.figure(figsize = (5,4))
		 sn.heatmap(df_cm, annot=True, annot_kws={"size": 10},cmap=plt.cm.Blues)
		 
		 plt.show()

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

def metrics_combined_graph(comb_y_true,comb_y_pred,comb_y_probs):
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
 
	
	x = ["2012", "2013", "2014", "2015"] 
	plt.plot(x,roc_l,label='ROC', linestyle = '--', color = 'black')
	plt.plot(x,f1_score_l,label='F1_score', linestyle = '-', color = 'gray')
	plt.legend()
	plt.ylabel('ROC/F1score')
	plt.show()

	

def logisticRegression_Tuning(list_years):
	

	print("Logistic Regression Tuning: ")
	print("Time Series Split  :")

	roc_cum=0
	C = np.logspace(-100, 10, 200)
	maxStat = 0.5
	tunedC = 1

	for c in C:
		total =[]
		for i in range(4):
			frames=[]
			for j in range(i+1):
					frames.append(list_years[j])
			train = pd.concat(frames)
			test = list_years[i+1]
			X_train = train.drop(['winner'],axis=1)
			scaler = StandardScaler()
			scaler.fit(X_train)
			X_train = scaler.transform(X_train)
			y_train = train['winner']
			X_test = test.drop(['winner'],axis=1)
			X_test = scaler.transform(X_test)
			y_test = test['winner']
			LR = LogisticRegression(C=c,random_state=0,max_iter=1000, class_weight='balanced').fit(X_train, y_train)
			y_preds = LR.predict(X_test)
			cf = confusion_matrix(y_test,y_preds)
			roc = roc_auc_score(y_test,y_preds)
			roc_cum +=roc
			total.append(roc)
		# To find C value for the highest maxstat
		if (np.mean(total))> maxStat:
				maxStat = np.mean(total)
				tunedC = c
	print("FINAL Results :")   
	print("C : " + str(tunedC))
	return tunedC

def logisticRegression(list_years):

	
	comb_y_true =[]
	comb_y_pred =[]
	comb_y_probs =[]
	print("Logistic Regression: ")

	total =0
	# Tuned value for c
	c =  logisticRegression_Tuning(list_years)
	for i in range(4):
		frames=[]
		for j in range(i+1):
				frames.append(list_years[j])
		train = pd.concat(frames)
		test = list_years[i+1]
		X_train = train.drop(['winner'],axis=1)
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		y_train = train['winner']
		X_test = test.drop(['winner'],axis=1)
		X_test = scaler.transform(X_test)
		y_test = test['winner']
		comb_y_true.append(np.array(y_test))
		LR = LogisticRegression(C =c,random_state=0,max_iter=100,class_weight='balanced').fit(X_train, y_train)
		y_preds = LR.predict(X_test)
		comb_y_pred.append(np.array(y_preds))
		y_score = LR.predict_proba(X_test)
		comb_y_probs.append(np.array(y_score[:,1]))
		#print("{} ROC is {}".format(2012+i,roc))
		print("Metrics for {} test set".format(2012+i))
		metrics(y_test,y_preds,y_score[:,1])

	# Mean Accuracy    
	metrics_combined(comb_y_true,comb_y_pred,comb_y_probs)



def DecisionTree_Params(list_years):

	print("Decision Tree Params: ")

	Min_samples_leaf = [20]
	Max_depth = [5,10]
	maxStat =0.5 

	final_samples = 0
	final_depth=0
	for samples in Min_samples_leaf:
		for depth in Max_depth:
				total =0
				total_prob =0
				for i in range(4):
					frames=[]
					for j in range(i+1):
						frames.append(list_years[j])
					train = pd.concat(frames)
					test = list_years[i+1]
					X_train = train.drop(['winner'],axis=1)
					scaler = StandardScaler()
					scaler.fit(X_train)
					X_train = scaler.transform(X_train)
					y_train = train['winner']
					X_test = list_years[i+1].drop(['winner'], axis = 1)
					X_test = scaler.transform(X_test)
					y_test = list_years[i+1]['winner']
					DT = DecisionTreeClassifier(criterion='gini', splitter='random',random_state=0,min_samples_leaf=samples,max_depth=depth,ccp_alpha=0.01)
					DT.fit(X_train,y_train)
					y_preds = DT.predict(X_test)
					y_score = DT.predict_proba(X_test)
					cf = confusion_matrix(y_test,y_preds)
					roc_prob = roc_auc_score(y_test,y_score[:,1])
					total_prob += roc_prob

				if((total_prob/4)>maxStat):
						final_samples = samples
						final_depth= depth

		return  final_samples,final_depth

def DecisionTree(list_years):

	comb_y_true =[]
	comb_y_pred =[]
	comb_y_probs =[]

	print("Decision Tree: ")
	samples,depth = DecisionTree_Params(list_years)
	print("Parameters chosen: min_samples_leaf as {} and max_depth as {}".format(samples,depth))
	total =0
	for i in range(4):
		frames=[]
		for j in range(i+1):
				frames.append(list_years[j])
		train = pd.concat(frames)
		test = list_years[i+1]
		X_train = train.drop(['winner'],axis=1)
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		y_train = train['winner']
		X_test = list_years[i+1].drop(['winner'], axis = 1)
		X_test = scaler.transform(X_test)
		y_test = list_years[i+1]['winner']
		DT = DecisionTreeClassifier(criterion='gini', splitter='random',random_state=0,min_samples_leaf=samples,max_depth=depth,ccp_alpha=0.01)
		DT.fit(X_train,y_train)
		comb_y_true.append(np.array(y_test))
		y_preds = DT.predict(X_test)
		comb_y_pred.append(np.array(y_preds))
		y_score = DT.predict_proba(X_test)
		comb_y_probs.append(np.array(y_score[:,1]))
		print("Metrics for {} test set".format(2011+i))
		metrics(y_test,y_preds,y_score[:,1])
	metrics_combined(comb_y_true,comb_y_pred,comb_y_probs)


def ANN(list_years):

	model_nn = keras.Sequential([
	keras.layers.Dense(units=15,input_dim=10,activation='relu'),
	keras.layers.Dense(units=25,activation='relu'),
	keras.layers.Dense(units=1,activation='sigmoid')])

	comb_y_true =[]
	comb_y_pred =[]
	comb_y_probs =[]

	# 2 hidden layers
	model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	print("ANN Neural Network:")
	total =0
	for i in range(4):
		frames=[]
		for j in range(i+1):
				frames.append(list_years[j])
		train = pd.concat(frames)
		test = list_years[i+1]
		X_train = train.drop(['winner'],axis=1)
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		y_train = train['winner']
		X_test = list_years[i+1].drop(['winner'], axis = 1)
		X_test = scaler.transform(X_test)
		y_test = list_years[i+1]['winner']

		history = model_nn.fit(X_train,y_train,epochs=100,shuffle=False,verbose=0)
		#y_preds = model_nn.predict(X_test)
		y_score = model_nn.predict(X_test)
		y_preds=y_score
		#y_preds = model_nn.predict(X_train)
		for i in range(len(y_preds)):
				if y_preds[i]>=0.5:
						y_preds[i]=1
				else:
						y_preds[i]=0
		
		#metrics(y_test,y_preds,y_score)
		keras.backend.clear_session()
		comb_y_true.append(np.array(y_test))
		comb_y_pred.append(np.array(y_preds))
		comb_y_probs.append(np.array(y_score))

	for i in range(4):
		print("Metrics for {} test set".format(2012+i))
		metrics(comb_y_true[i],comb_y_pred[i],comb_y_probs[i])  
	metrics_combined(comb_y_true,comb_y_pred,comb_y_probs)


d_2011 = pd.read_csv('../Data/teamBal_2011.csv')
d_2012 = pd.read_csv('../Data/teamBal_2012.csv')
d_2013 = pd.read_csv('../Data/teamBal_2013.csv')
d_2014 = pd.read_csv('../Data/teamBal_2014.csv')
d_2015 = pd.read_csv('../Data/teamBal_2015.csv')

print("{} shape: {}".format(2011,d_2011.shape))
print("{} shape: {}".format(2012,d_2012.shape))
print("{} shape: {}".format(2013,d_2013.shape))
print("{} shape: {}".format(2014,d_2014.shape))
print("{} shape: {}".format(2015,d_2015.shape))


# Features considered for team data
feature_cols = ['AttackPCT','AssistPCT','ServePCT','ReceptPCT','Set_win_ratio',
								'AttackPCT.1','AssistPCT.1','ServePCT.1','ReceptPCT.1','Set_win_ratio.1',
								'winner']

d_2011 = d_2011[feature_cols]
d_2012 = d_2012[feature_cols]
d_2013 = d_2013[feature_cols]
d_2014 = d_2014[feature_cols]
d_2015 = d_2015[feature_cols]

list_years = [d_2011,d_2012,d_2013,d_2014,d_2015]

print("Logistic Regression:")
logisticRegression(list_years)
print()
print("Decision Tree:")
DecisionTree(list_years)
print()
print("Artificial Neural Network:")
ANN(list_years)
print()

