import pandas as pd
conc=pd.read_csv('concrete.csv')
conc=conc[conc.columns[1:10]]
y=conc['strength']
X=conc.drop(['strength'],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=54,shuffle=True)
from sklearn.ensemble import ExtraTreesRegressor
clf=ExtraTreesRegressor(n_estimators=382, max_depth=None,min_samples_split=2,random_state=7)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
dic=dict(zip(X.columns,clf.feature_importances_))
for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))
