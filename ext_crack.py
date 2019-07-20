from numpy import genfromtxt
cd=genfromtxt('cd_gray.csv',delimiter=',')

import numpy as np
cdy=np.ones(2025)
udy=np.zeros(2025)
ud=genfromtxt('ud_gray.csv',delimiter=',')
X=np.concatenate((cd,ud))
y=np.concatenate((cdy,udy))
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=90,shuffle=True)
from sklearn.ensemble import ExtraTreesClassifier
clf=ExtraTreesClassifier(n_estimators=382, n_jobs=-1,max_depth=None,min_samples_split=2,random_state=90)
clf.fit(X_train,y_train)
print("%.4f"%clf.score(X_test,y_test))
