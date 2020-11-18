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
  tunedParameters['epochs'] = 50
  tunedParameters['optimizer'] = 'adam'




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
      y_preds = model_nn.predict(X_test)
      #y_preds = model_nn.predict(X_train)
      for i in range(len(y_preds)):
          if y_preds[i]>=0.5:
              y_preds[i]=1
          else:
              y_preds[i]=0
      cf_l.append(confusion_matrix(y_test,y_preds))
      roc = roc_auc_score(y_test,y_preds)
      roc_l.append(roc)
      total += roc

  for i in range(4):
    print("{} ROC is {}".format(2012+i,roc_l[i]))
    print(cf_l[i])
  print("Average: {}".format(total/4))

if __name__ == "__main__":
  main()