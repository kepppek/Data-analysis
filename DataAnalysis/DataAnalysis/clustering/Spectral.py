import pandas as pd
from sklearn.cluster import SpectralClustering
import numpy as np
from enum import Enum

class AssignLabels(Enum):
    kmeans = "kmeans"
    discretize = "discretize"
    cluster_qr = "cluster_qr"


def Spectral(table, parametrs):
    clustering = SpectralClustering(n_clusters=parametrs[0], assign_labels=parametrs[1], random_state=parametrs[2]).fit(table)
    Y_preds = pd.DataFrame(data = clustering.labels_, columns = ["Прогноз"])
    return Y_preds
