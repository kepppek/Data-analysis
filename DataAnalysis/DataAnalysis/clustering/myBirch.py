import numpy as np
from enum import Enum
import pandas as pd
from sklearn.cluster import Birch



def myBirch(table, parametrs):    
    #brc = Birch(n_clusters = parametrs[0]).fit(table)
    #Y_preds = pd.DataFrame(data = brc.labels_, columns = ["Y_Pred"])
    #X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
    table = np.ascontiguousarray(table)
    #table = np.array(table, np.double)
    brc = Birch(n_clusters = parametrs[0])
    brc.fit_predict(table)
    #brc.predict(table)
    Y_preds = pd.DataFrame(data = brc.labels_, columns = ["Прогноз"])
    return Y_preds
