from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def gaussianProcessClassifier(table,target,parametrs):
    x_train, x_test, y_train, y_test = train_test_split(table, target, test_size=parametrs[0])
    gpc = GaussianProcessClassifier(max_iter_predict=parametrs[1], n_restarts_optimizer = parametrs[2])
    gpc.fit(x_train, y_train)
    y_pred = gpc.predict(x_test)
    return  x_test.assign(Истина=y_test,Прогноз=y_pred)
