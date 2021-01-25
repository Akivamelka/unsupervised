# Pre procesing of the second data set for clothing.
# All indices for the file are therfore 2.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Opening the data file.
data_set = pd.read_csv(r"e-shop clothing 2008.csv", delimiter=";")

# We define the class files and remove the features from the data.
# we also save them for further use.
target = data_set.loc[:,"country"]
target = target.astype('category')
target = target.cat.codes
target.to_csv("target_2.csv", index=False)

# We remove the "price" for now as it is numerical and will add it later.
price = data_set.loc[:,"price"]

# After analysis, we remove some features.
data_set.drop(["year", "country", "page 2 (clothing model)", "session ID", "price"], axis=1, inplace=True)

# Categorical features that need to be ordered.
data_set.month = pd.Categorical(data_set.month, ordered=True)
data_set.day = pd.Categorical(data_set.day, ordered=True)
data_set.order = pd.Categorical(data_set.order, ordered=True)
data_set.page = pd.Categorical(data_set.page, ordered=True)

for col in data_set.columns:
    data_set[col] = data_set[col].astype('category')
    data_set[col] = data_set[col].cat.codes

# WE reinsert "price"
data_set = pd.concat([data_set, price], axis=1)

# Scaling of the data.
std = MinMaxScaler()
data_set_scaled = std.fit_transform(data_set)
data_set_scaled_df = pd.DataFrame(data_set_scaled, columns=data_set.columns)

# Save to csv files.
data_set_scaled_df.to_csv("data_preprocess_scaled_2.csv", index=False)
data_set.to_csv("data_preprocess_2.csv", index=False)