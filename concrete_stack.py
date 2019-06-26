import pandas as pd
import numpy as np
conc=pd.read_csv('concrete.csv')
from sklearn.model_selection import KFold
y=np.array(conc['csMPa'])
X=np.array(conc.drop(['csMPa'],axis=1))
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from mlxtend.regressor import StackingRegressor
rf=RandomForestRegressor(n_estimators=682, max_depth=None,random_state=8)
ext=ExtraTreesRegressor(n_estimators=682,min_samples_split=2,random_state=8)
clf=StackingRegressor(regressors=[ext],meta_regressor=rf)

scores=[]
for train,test in kfold.split(X,y):
 clf.fit(X[train], y[train])
 score=clf.score(X[test], y[test])
 print(score)
 scores.append(score)
print("%.3f%% (+/- %.3f)" % (np.mean(scores),np.std(scores)))

