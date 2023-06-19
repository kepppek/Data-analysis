import numpy as np
from enum import Enum
import pandas as pd
from sklearn.cluster import KMeans

    
def K_Means(table, parametrs):
    kmeans = KMeans(n_clusters=parametrs[0], random_state=parametrs[1]).fit(table) 
    Y_preds = pd.DataFrame(data = kmeans.labels_, columns = ["Прогноз"])
    return Y_preds

