# Pre procesing of the first data set for diabetes.
# All indices for the file are therfore 1.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Opening the data file.
data_set = pd.read_csv(r"HTRU.csv", delimiter=',')

# We define the class files and remove the features from the data.
# we also save them for further use.
target = data_set.loc[:,"Class"]
data_set = data_set.drop(["Class"], axis=1)
target.to_csv("target_5.csv", index=False)

#We removed features
data_set = data_set.drop(["X5","X6","X7","X8"], axis=1)

# We scale them so that all features have the same influnece on the clustering.
#scale = MinMaxScaler()
scale = StandardScaler()
data_scaled = scale.fit_transform(data_set)
data_scaled_df = pd.DataFrame(data_scaled, columns=data_set.columns)

#data = data_scaled_df.copy()
data = data_set.copy()
data.to_csv("data_preprocess_5.csv", index=False)