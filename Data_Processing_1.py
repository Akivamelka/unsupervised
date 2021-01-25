# Pre procesing of the first data set for diabetes.
# All indices for the file are therfore 1.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Opening the data file.
data_set = pd.read_csv(r"diabetic_data.csv", na_values=["?", "Unknown/Invalid"], low_memory=False)

# After analysis of the data, we observed that 3 features were missing a lot of data.
# So we removed them. Also the "encounter_id"is unique for all inmstances
# so it does not provide any info fro the clustering.
data_set.drop(['weight', 'medical_specialty', 'payer_code', 'encounter_id'], axis=1, inplace=True)

# We also remove instances with missing data.
data_set.dropna(inplace=True)
data_set.reset_index(drop=True, inplace=True)

# A lot of instances are from the same patient.
# Since we want independant instances and the clustering not to be biased,
# we remoce dupliocates.
data_set.drop_duplicates(subset='patient_nbr', inplace=True)
data_set.reset_index(drop=True, inplace=True)

# We define the class files and remove the features from the data.
# we also save them for further use.
target1 = data_set.loc[:,"gender"]
target1 = target1.astype('category')
target1 = target1.cat.codes
target1.to_csv("target_1_1.csv", index=False)

target2 = data_set.loc[:,"race"]
target2 = target2.astype('category')
target2 = target2.cat.codes
target2.to_csv("target_1_2.csv", index=False)

data_set.drop(['race', 'gender', 'patient_nbr'], axis=1, inplace=True)

# We seperate features into numerical and categorical.
# For the 'age' we keep the order since it is relevant.
# After analysis, we removed all the columns of medication
# since only a very small number of patients actually used them.
numerical_columns = ['time_in_hospital', 'num_lab_procedures', 'num_medications',
                        'num_procedures', 'number_diagnoses', 'number_emergency',
                        'number_inpatient', 'number_outpatient']
categorical_columns = ['age', 'admission_type_id', 'diag_1', 'diag_2', 'diag_3',
                       'discharge_disposition_id','admission_source_id', 'insulin',
                       'change', 'diabetesMed', 'readmitted']

numerical_df = data_set[numerical_columns]
categorical_df = data_set[categorical_columns]
categorical_df.age = pd.Categorical(categorical_df.age, ordered=True)

# The categorical features have to be replaced with values.
# Except for age, the order is not relevant.
for col in categorical_df.columns:
    categorical_df[col] = categorical_df[col].astype('category')
    categorical_df[col] = categorical_df[col].cat.codes

# We scale them so that all features have the same influnece on the clustering.
std_num = MinMaxScaler()
numerical_scaled = std_num.fit_transform(numerical_df)
numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_df.columns)

std_cat = MinMaxScaler()
categorical_scaled = std_cat.fit_transform(categorical_df)
categorical_scaled_df = pd.DataFrame(categorical_scaled, columns=categorical_df.columns)

# Finally the data is saved.
# We save the scaled data but also the non scaled for comparison purposes.
data_scaled = pd.concat([numerical_scaled_df, categorical_scaled_df], axis=1)
data_scaled.reset_index(drop=True, inplace=True)
data_scaled.to_csv("data_preprocess_scaled_1.csv", index=False)

data = pd.concat([numerical_df, categorical_df], axis=1)
data.reset_index(drop=True, inplace=True)
data.to_csv("data_preprocess_1.csv", index=False)
