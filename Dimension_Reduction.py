import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

suffix = "scaled_3"
kpca = True

class Dim_red_PCA(object):

    def __init__(self):
        pass

    def view_variance(self, data, suffix):
        file = f"pca_variance_{suffix}.png"
        plt.figure(figsize=(8, 8))
        pca = PCA()
        pca.fit(data)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.text(0.5, 0.85, '95% cut-off threshold', color='red', fontsize=16)
        plt.axhline(y=0.95, color='red', linestyle='-')
        plt.savefig(file)
        # plt.show()

    def threshold_to_components(self, data, thresh=0.95):
        pca = PCA(n_components=thresh).fit(data)
        print(f"number of components for {thresh}% of variance is {pca.n_components_}")
        return pca.n_components_

    def components_to_threshold(self, data, n):
        pca = PCA(n_components=n).fit(data)
        print(f"For {n} components, the variance is {sum(pca.explained_variance_ratio_) * 100}%")

    def dimension_reduction(self, data, n, suffix):
        file = f"pca_dim{n}_{suffix}.csv"
        pca = PCA(n_components=n)
        x_pca = pca.fit_transform(data)
        new_frame = pd.DataFrame(x_pca)
        new_frame.to_csv(file, index=False)

class Dim_red_KPCA(object):

    def __init__(self):
        pass

    def dimension_reduction(self, data, n, suffix):
        file = f"kpca_dim{n}_{suffix}.csv"
        kpca = KernelPCA(n_components=3, kernel='rbf')
        kx_pca = kpca.fit_transform(data)
        new_frame = pd.DataFrame(kx_pca)
        new_frame.to_csv(file, index=False)

class Dim_red_tSNE(object):

    def __init__(self):
        pass

    def dimension_reduction(self, data, suffix):
        file = f"tsne_dim2_{suffix}.csv"
        tsne = TSNE(n_components=2, verbose=True, perplexity=40, n_iter=300)
        x_tsne = tsne.fit_transform(data)
        new_frame = pd.DataFrame(x_tsne)
        new_frame.to_csv(file, index=False)

class Viewer(object):

    def __init__(self):
        pass

    def simple_viewer(self, data, suffix1, suffix2):
        file = f"graph_dim2_{suffix2}_{suffix1}.png"
        plt.figure(figsize=(8, 8))
        sns.scatterplot(
            x=data.values[:, 0], y=data.values[:, 1],
            alpha=0.3)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(file)
        # plt.show()

    def view_vs_target(self, data, target, suffix1, suffix2):
        file = f"graph_dim2_{suffix2}_{suffix1}.png"
        df_subset = pd.DataFrame({"Dim-One": data.values[:, 0], "Dim-two": data.values[:, 1]})
        df_subset['clusters'] = target

        plt.figure(figsize=(8, 8))
        sns.scatterplot(
            x="Dim-One", y="Dim-two",
            hue="clusters",
            palette=sns.color_palette("hls", df_subset['clusters'].nunique()),
            data=df_subset,
            legend="full",
            alpha=0.3
        )
        plt.xticks([])
        plt.yticks([])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(file)
        # plt.show()

if __name__ == "__main__":

    # Open original data file
    data = pd.read_csv(f"data_preprocess_{suffix}.csv", delimiter=",")
    view_tool = Viewer()

    # Variance analysis of the PCA components.
    first_pca = Dim_red_PCA()
    n_threshold = first_pca.threshold_to_components(data, 0.95)
    first_pca.view_variance(data, suffix)

    # Dimension reduction with PCA
    first_pca.dimension_reduction(data, n_threshold, suffix)

    # Dimension reduction for visualization with PCA
    first_pca.dimension_reduction(data, 2, suffix)
    data_pca_dim2 = pd.read_csv(f"pca_dim2_{suffix}.csv", delimiter=",")
    view_tool.simple_viewer(data_pca_dim2, suffix, "pca")

    # Dimension reduction for visualization with tSNE
    first_tsne = Dim_red_tSNE()
    first_tsne.dimension_reduction(data, suffix)
    data_tsne_dim2 = pd.read_csv(f"tsne_dim2_{suffix}.csv", delimiter=",")
    view_tool.simple_viewer(data_tsne_dim2, suffix, "tsne")

    if kpca:
        # Dimension reduction with PCA
        first_kpca = Dim_red_KPCA()
        first_kpca.dimension_reduction(data, n_threshold, suffix)

        # Dimension reduction for visualization with PCA
        first_kpca.dimension_reduction(data, 2, suffix)
        data_kpca_dim2 = pd.read_csv(f"kpca_dim2_{suffix}.csv", delimiter=",")
        view_tool.simple_viewer(data_kpca_dim2, suffix, "kpca")
