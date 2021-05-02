from Dimension_Reduction import Viewer
import pandas as pd

view_tool = Viewer()
reduc = 'pca'
suffix = '5'
data_plot = pd.read_csv(f"{reduc}_dim2_{suffix}.csv", delimiter=",")
models = ['km', 'fuzz', 'gmm', 'dbsc', 'hier', 'spec' ]

for model in models:
    print(model)
    labels = pd.read_csv(f"labels_{model}_{suffix}.csv", delimiter=",")
    view_tool.view_vs_target(data_plot, labels, suffix, model)