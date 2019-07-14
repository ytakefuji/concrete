import pandas as pd
conc=pd.read_excel('Concrete_Data.xls')
conc.to_csv('test.csv')
with open("test.csv",'r') as f, open("newtest.csv",'w') as f1:
 next(f) 
 for line in f:
  f1.write(line)
conc=pd.read_csv('newtest.csv')
conc.columns=[',','cement','blast','ash','water','sp','cagg','fagg','age','strength']
conc[conc.columns[1:10]].to_csv('concrete.csv')
y=conc['strength']
X=conc.drop(['strength',','],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=54,shuffle=True)
from sklearn.ensemble import RandomForestRegressor
clf=RandomForestRegressor(n_estimators=382, max_depth=None,min_samples_split=2,random_state=7)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
dic=dict(zip(X.columns,clf.feature_importances_))
for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))
