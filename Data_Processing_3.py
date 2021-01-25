# Pre procesing of the third data set for shoppers intention.
# All indices for the file are therfore 3.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Opening the data file.
data_set = pd.read_csv(r"online_shoppers_intention.csv", na_values=["?", "Unknown/Invalid"], low_memory=False)

# We define the class files and remove the features from the data.
# we also save them for further use.
target1 = data_set.loc[:,"Revenue"]
target1 = target1.astype('category')
target1 = target1.cat.codes
target1.to_csv("target_3_1.csv", index=False)

target2 = data_set.loc[:,"VisitorType"]
target2 = target2.astype('category')
target2 = target2.cat.codes
target2.to_csv("target_3_2.csv", index=False)

target3 = data_set.loc[:,"Weekend"]
target3 = target3.astype('category')
target3 = target3.cat.codes
target3.to_csv("target_3_3.csv", index=False)

# We seperate features into numerical and categorical.
# For the numerical features, we seperate them whether they are duration or not.
categorical_features = ['OperatingSystems','Browser', 'Region', 'TrafficType']
categorical_df = data_set[categorical_features]

numerical_duration_features = ['Administrative_Duration','Informational_Duration', 'ProductRelated_Duration']
numerical_duration_features_df = data_set[numerical_duration_features]

numerical_features = ['Administrative','Informational', 'ProductRelated']
numerical_features_df = data_set[numerical_features]

numerical_features_2 = ['BounceRates','ExitRates', 'PageValues', 'SpecialDay']
numerical_features_2_df = data_set[numerical_features_2]

# We scale them so that all features have the same influnece on the clustering.
std_num_dur = MinMaxScaler()
numerical_duration_scaled = std_num_dur.fit_transform(numerical_duration_features_df)
numerical_duration_scaled_df = pd.DataFrame(numerical_duration_scaled, columns=numerical_duration_features_df.columns)

std_num = MinMaxScaler()
numerical_scaled_2 = std_num.fit_transform(numerical_features_2_df)
numerical_scaled_2_df = pd.DataFrame(numerical_scaled_2, columns=numerical_features_2_df.columns)

# After analysis, we observe that the feature with duration are extremely correlated
# to their duration, so we removed those features and kept only the durations.
data_scaled = pd.concat([numerical_duration_scaled_df, numerical_scaled_2_df, categorical_df], axis = 1)
data_scaled.reset_index(drop=True, inplace=True)

# Finally the data is saved.
data_scaled.to_csv("data_preprocess_scaled_3.csv", index=False)
