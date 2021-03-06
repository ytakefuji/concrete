from numpy import genfromtxt
cd=genfromtxt('cd_gray.csv',delimiter=',')
print('loading cd_gray.csv is completed.')

import numpy as np
np.random.seed(90)
cdy=np.ones(2025)
udy=np.zeros(2025)
ud=genfromtxt('ud_gray.csv',delimiter=',')
print('loading ud_gray.csv is completed.')
X=np.concatenate((cd,ud))
y=np.concatenate((cdy,udy))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=90,shuffle=True)
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'n_jobs':-1,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=80,
                valid_sets=lgb_eval,
                verbose_eval=1,
                early_stopping_rounds=5)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
from sklearn.metrics import accuracy_score
pred=[]
for i in y_pred:
 if i>0.5: pred.append(1)
 else:pred.append(0)
y_pred=pred
print("%.4f"%accuracy_score(y_test,y_pred))
