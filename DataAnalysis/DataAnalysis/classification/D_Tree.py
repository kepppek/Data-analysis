from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def D_Tree(table,target,parametrs):
    x_train, x_test, y_train, y_test = train_test_split(table, target, test_size=parametrs[0])
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)
    #y_pred = cross_val_score(clf, table, target, cv=10)
    y_pred = clf.predict(x_test)
    #return [x_test,y_pred,y_test]
    return x_test.assign(Истина=y_test,Прогноз=y_pred)
