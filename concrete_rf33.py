import pandas as pd
conc=pd.read_excel('Concrete_Data.xls')
conc.to_csv('test.csv')
with open("test.csv",'r') as f, open("newtest.csv",'w') as f1:
 next(f) 
 for line in f:
  f1.write(line)
conc=pd.read_csv('newtest.csv')
conc.columns=['num','cement','blast','ash','water','sp','cagg','fagg','age','strength']
from sklearn.model_selection import train_test_split
y=conc['strength']
X=conc.drop(['strength','num'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=54,shuffle=True)
from sklearn.ensemble import RandomForestRegressor
clf=RandomForestRegressor(n_estimators=382, max_depth=None,min_samples_split=2,random_state=8)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(clf.score(X_test,y_test))
dic=dict(zip(X.columns,clf.feature_importances_))
for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))
tree=clf.estimators_[5]
from sklearn.tree import export_graphviz
# Export as dot file
import pydotplus
from io import StringIO
dotfile=StringIO()
export_graphviz(tree, out_file=dotfile) 
graph=pydotplus.graph_from_dot_data(dotfile.getvalue())
graph.write_png("tree.png")
from PIL import Image
image=Image.open('tree.png')
image.show()
