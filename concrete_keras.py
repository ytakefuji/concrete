from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import random as rn
from sklearn.model_selection import train_test_split
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
conc=pd.read_csv('concrete.csv')
conc=conc[conc.columns[1:10]]
Y=conc['strength'].values
X=conc.drop(['strength'],axis=1).values
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
model = Sequential()
model.add(Dense(540,input_dim=8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
model.fit(x_train,y_train,epochs = 32,batch_size=32,validation_data=(x_test,y_test),verbose=2)
y_predict=model.predict(x_test)
from sklearn.metrics import r2_score
print(r2_score(y_test,y_predict))
