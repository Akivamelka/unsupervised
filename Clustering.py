import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Dimension_Reduction import Viewer

from Silhouette import Silhouette_viewer

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import skfuzzy as fuzz
from sklearn.ensemble import IsolationForest

suffix = "5"
data = pd.read_csv(f"data_preprocess_{suffix}.csv", delimiter=",")
data_plot = pd.read_csv(f"pca_dim2_{suffix}.csv", delimiter=",")
n_clusters = 2

# Compute Elbow method with K-Means to determine number of clusters.
def compute_elbow(max_num_cluster=7):
    file = f"elbow_{suffix}.png"
    distortions = []
    K = range(1, max_num_cluster)
    for k in K:
        print("K-Means with k = ", k)
        kmeanModel = KMeans(n_clusters=k, random_state=10)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
    plt.figure(figsize=(8, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig(file)

def compute_silhouette_profile():
    for nn in range(2,4):
        model = KMeans(n_clusters=nn, random_state=10)
        labels = model.fit_predict(data)
        centers = model.cluster_centers_
        sil = Silhouette_viewer()
        sil.silhouette_plot(data, labels, centers, f'silhouette_{suffix}_{nn}.png')

def clustering():

    view_tool = Viewer()

    print("kmeans")
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    labels_km = kmeans.fit_predict(data)
    labels_km_df = pd.DataFrame(labels_km)
    labels_km_df.to_csv(f"labels_km_{suffix}.csv", index=False)
    view_tool.view_vs_target(data_plot, labels_km_df, suffix, 'km')

    print("fuzz")
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T.values, n_clusters, 2, error=0.005, maxiter=1000)
    labels_fuz = np.argmax(u, axis=0)
    labels_fuz_df = pd.DataFrame(labels_fuz)
    labels_fuz_df.to_csv(f"labels_fuzz_{suffix}.csv", index=False)
    view_tool.view_vs_target(data_plot, labels_fuz_df, suffix, 'fuzz')

    print("gmm")
    gmm = GMM(n_components=n_clusters, random_state=10)
    labels_gmm = gmm.fit_predict(data)
    labels_gmm_df = pd.DataFrame(labels_gmm)
    labels_gmm_df.to_csv(f"labels_gmm_{suffix}.csv", index=False)
    view_tool.view_vs_target(data_plot, labels_gmm_df, suffix, 'gmm')

    print("dbsc")
    dbscan = DBSCAN(eps=2, min_samples=20).fit(data)
    labels_dbsc = dbscan.labels_
    labels_dbsc_df = pd.DataFrame(labels_dbsc)
    labels_dbsc_df.to_csv(f"labels_dbsc_{suffix}.csv", index=False)
    view_tool.view_vs_target(data_plot, labels_dbsc_df, suffix, 'dbsc')

    print("hier")
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical.fit(data)
    labels_hier = hierarchical.labels_
    labels_hier_df = pd.DataFrame(labels_hier)
    labels_hier_df.to_csv(f"labels_hier_{suffix}.csv", index=False)
    view_tool.view_vs_target(data_plot, labels_hier_df, suffix, 'hier')

    print("spec")
    spectral = SpectralClustering(n_clusters=n_clusters)
    spectral.fit(data)
    labels_spec = spectral.labels_
    labels_spec_df = pd.DataFrame(labels_spec)
    labels_spec_df.to_csv(f"labels_spec_{suffix}.csv", index=False)
    view_tool.view_vs_target(data_plot, labels_spec_df, suffix, 'spec')

    print("islf")
    islF_model = IsolationForest(random_state=0, contamination=0.08)
    labels_islf = islF_model.fit_predict(data)
    labels_islf_df = pd.DataFrame(labels_islf)
    labels_islf_df.to_csv(f"labels_islf_{suffix}.csv", index=False)
    view_tool.view_vs_target(data_plot, labels_islf_df, suffix, 'islf')

compute_elbow()
compute_silhouette_profile()
clustering()

