from enum import Enum
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class KNNAlgorithmType(Enum):
    auto = 'auto'
    ball_tree = 'ball_tree'
    kd_tree = 'kd_tree'
    brute = 'brute'


class KNNWeightType(Enum):
    uniform = 'uniform'
    distance = 'distance'


class KNNMetricsType(Enum):
    euclidean = 'euclidean'
    minkowski = 'minkowski'
    manhattan = 'manhattan'
    chebyshev = 'chebyshev'


def Nearest_Neighbors(table,target,parametrs):
     x_train, x_test, y_train, y_test = train_test_split(table, target, test_size=parametrs[3])
     classifier = KNeighborsClassifier(n_neighbors=parametrs[0],algorithm=parametrs[1],weights = parametrs[2])
     classifier.fit(x_train, y_train)
     y_pred = classifier.predict(x_test)
     return  x_test.assign(Истина=y_test,Прогноз=y_pred)
 # return  x_test.assign(Y_True=y_test,Y_Pred=y_pred)
    
