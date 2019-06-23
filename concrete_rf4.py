import pandas as pd
import numpy as np
conc=pd.read_excel('Concrete_Data.xls')
conc.to_csv('test.csv')
with open("test.csv",'r') as f, open("newtest.csv",'w') as f1:
 next(f) 
 for line in f:
  f1.write(line)
conc=pd.read_csv('newtest.csv')
conc.columns=['num','cement','blast','ash','water','sp','cagg','fagg','age','strength']
from sklearn.model_selection import KFold
y=np.array(conc['strength'])
X=np.array(conc.drop(['strength','num'],axis=1))
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
from sklearn.ensemble import RandomForestRegressor
clf=RandomForestRegressor(n_estimators=382, max_depth=None,random_state=8)
scores=[]
for train,test in kfold.split(X,y):
 clf.fit(X[train], y[train])
 score=clf.score(X[test], y[test])
 print(score)
 scores.append(score)
print("%.3f%% (+/- %.3f)" % (np.mean(scores),np.std(scores)))

