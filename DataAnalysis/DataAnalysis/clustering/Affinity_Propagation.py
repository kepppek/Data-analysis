from enum import Enum
import pandas as pd

from sklearn.cluster import AffinityPropagation

class Affinity(Enum):
    euclidean = "euclidean"
    precomputed = "precomputed"

def Affinity_Propagation(table, parametrs):
    clustering = AffinityPropagation(affinity = parametrs[0], random_state=parametrs[1], preference = parametrs[2], damping = parametrs[3], max_iter = parametrs[4])
    Y_preds = pd.DataFrame(data = clustering.fit_predict(table), columns = ["Прогноз"])
    return Y_preds
