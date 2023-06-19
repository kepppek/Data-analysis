from sklearn.cluster import MeanShift
import numpy as np
import pandas as pd


def Mean_Shift(table, parametrs):
    clustering = MeanShift(bandwidth=parametrs[0]).fit(table)
    Y_preds = pd.DataFrame(data = clustering.labels_, columns = ["Прогноз"])
    return Y_preds