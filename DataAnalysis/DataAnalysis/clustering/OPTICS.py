from enum import Enum
import pandas as pd

from sklearn.cluster import OPTICS


def Optics(table, parametrs):
    clustering = OPTICS(min_samples = parametrs[0] , xi = parametrs[1] , min_cluster_size = parametrs[2]).fit(table)  
    Y_preds = pd.DataFrame(data = clustering.labels_, columns = ["Прогноз"])
    return Y_preds