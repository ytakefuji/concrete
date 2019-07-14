#run python3
import pandas as pd
import numpy as np
import random as rn
conc=pd.read_csv('concrete.csv')
conc=conc[conc.columns[1:10]]
from sklearn.model_selection import KFold
y=np.array(conc['strength'])
X=np.array(conc.drop(['strength'],axis=1))
kfold = KFold(n_splits=10, shuffle=True, random_state=43)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
K.floatx()
tf.set_random_seed(7)
config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=config)
K.set_session(sess)
np.random.seed(7)
rn.seed(7)

from mlxtend.regressor import StackingRegressor
rf=RandomForestRegressor(n_estimators=54, max_depth=None,random_state=7)
ext=ExtraTreesRegressor(n_estimators=284,min_samples_split=2,random_state=55)
def create_model():
 model = Sequential()
 model.add(Dense(64,input_dim=8, activation='relu'))
 model.add(BatchNormalization())
 model.add(Dense(20, activation='relu'))
 model.add(BatchNormalization())
 model.add(Dense(40, activation='relu'))
 model.add(BatchNormalization())
 model.add(Dense(30, activation='relu'))
 model.add(BatchNormalization())
 model.add(Dense(20, activation='relu'))
 model.add(Dense(1))
 model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
 return model

nn= KerasRegressor(build_fn=create_model, epochs=32, batch_size=32, verbose=0)
clf=StackingRegressor(regressors=[ext,nn],meta_regressor=rf)

scores=[]
for train,test in kfold.split(X,y):
 clf.fit(X[train], y[train])
 score=clf.score(X[test], y[test])
 print(score)
 scores.append(score)
print("%.3f%% (+/- %.3f)" % (np.mean(scores),np.std(scores)))

