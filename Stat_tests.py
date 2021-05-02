from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, ttest_ind

suffix = '5'
data = pd.read_csv(f"data_preprocess_{suffix}.csv", delimiter=",")
models = ['km', 'fuzz', 'gmm', 'dbsc', 'hier', 'islf' ]

labels_islf = pd.read_csv(f"labels_islf_{suffix}.csv", delimiter=",")
sil_values_islf = silhouette_samples(data, labels_islf)

for model in models:
    print(model)
    labels = pd.read_csv(f"labels_{model}_{suffix}.csv", delimiter=",")
    sil_values = silhouette_samples(data, labels)
    stat1, pv1 = ttest_ind(sil_values_islf, sil_values, equal_var=True)
    stat2, pv2 = ttest_ind(sil_values_islf, sil_values, equal_var=False)
    print(stat1, pv1)
    print(stat2, pv2)

sil = []
for model in models:
    print(model)
    labels = pd.read_csv(f"labels_{model}_{suffix}.csv", delimiter=",")
    sil_values = silhouette_samples(data, labels)
    sil.append(sil_values)
    score = np.mean(sil_values)
    print(model, score)

sil = np.array(sil)
f1, p1 = f_oneway(sil[0],sil[1],sil[2],sil[3],sil[4],sil[5])
f2, p2 = kruskal(sil[0],sil[1],sil[2],sil[3],sil[4],sil[5])
print(p1)
print(p2)