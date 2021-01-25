import numpy as np
import pandas as pd

from sklearn.metrics import silhouette_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import mutual_info_score

suffix = 'scaled_3'
data = pd.read_csv("data_preprocess_scaled_3.csv", delimiter=",")
target = pd.read_csv("target_3_1.csv", delimiter=",")
models = ['km', 'fuzz', 'gmm', 'dbsc', 'hier', 'spec' ]
results = []

for model in models:
    labels = pd.read_csv(f"labels_{model}_{suffix}.csv", delimiter=",")
    results.append([silhouette_score(data, labels, metric='cosine'),
                    fowlkes_mallows_score(target.values[:, 0], labels.values[:, 0]),
                    mutual_info_score(target.values[:, 0], labels.values[:, 0])])

results_df = pd.DataFrame(results)
results_df.to_csv(f"metrics_{suffix}.csv")