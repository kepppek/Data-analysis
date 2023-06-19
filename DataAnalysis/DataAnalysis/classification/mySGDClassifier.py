from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

def mySGDClassifier(table,target,parametrs):
    x_train, x_test, y_train, y_test = train_test_split(table, target, test_size=parametrs[0])
    clf = make_pipeline(StandardScaler(), 
                        SGDClassifier(max_iter=parametrs[1], tol=parametrs[2]))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    #return [x_test,y_pred,y_test]
    return x_test.assign(Истина=y_test,Прогноз=y_pred)
