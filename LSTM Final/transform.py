import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load Data
input_file = r'D:\School\ADMU\4Y\SEM 1\MATH 199.11\Final\DAILY\Mindanao\MIN_Daily_Complete.csv'
data = pd.read_csv(input_file).fillna(0)

def process_columns(df, value, substrings_ffill, substrings_interpolate):
    df.replace(value, np.nan, inplace=True)
    
    ffill_cols = df.filter(df.columns.str.contains('|'.join(substrings_ffill), case=False)).ffill()
    interpolate_cols = df.filter(df.columns.str.contains('|'.join(substrings_interpolate), case=False)).interpolate(method='linear')
    
    return ffill_cols, interpolate_cols

# Identify and Process Columns
columns_with_minus_999 = data.loc[:, (data == -999).any(axis=0)]
rainfall_cols, temp_cols_interpolated = process_columns(columns_with_minus_999, -999, 
                                                         substrings_ffill=['rainfall'], 
                                                         substrings_interpolate=['tmax', 'tmin'])
X = data.copy()
X.update(rainfall_cols)
X.update(temp_cols_interpolated)
y = data[['GWAP', 'LWAP']]

# Split Data
train_size = int(0.6 * len(X))
val_size = int(0.2 * len(X))
test_size = len(X) - train_size - val_size

train_data, val_data, test_data = np.split(X, [train_size, train_size + val_size])
train_labels, val_labels, test_labels = np.split(y, [train_size, train_size + val_size])

# Classify Features into Different Transformations
def classify_features(data):
    minmax_cols, boxcox_cols, yeojohnson_cols = [], [], []
    for column in data.columns:
        skewness = data[column].skew()
        kurtosis = data[column].kurtosis()
        is_positive = np.all(data[column] > 0)

        if -1 <= skewness <= 1 and -1 <= kurtosis <= 1:
            minmax_cols.append(column)
        elif is_positive:
            boxcox_cols.append(column)
        else:
            yeojohnson_cols.append(column)
    
    return minmax_cols, boxcox_cols, yeojohnson_cols

minmax_cols, boxcox_cols, yeojohnson_cols = classify_features(X)
minmax_colsy, boxcox_colsy, yeojohnson_colsy = classify_features(y)

# Scaling and Transforming Data
def fit_transform_pipeline(train_df, val_df, test_df, cols, method='box-cox'):
    pipeline = Pipeline([
        ('power_transformer', PowerTransformer(method=method, standardize=False)),
        ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))
    ])
    pipeline.fit(train_df[cols])
    train_transformed = pipeline.transform(train_df[cols])
    val_transformed = pipeline.transform(val_df[cols])
    test_transformed = pipeline.transform(test_df[cols])
    
    joblib.dump(pipeline, f'{method}_pipeline.pkl')
    return train_transformed, val_transformed, test_transformed

# Process MinMaxScaler for MinMax columns
minmax_scaler = MinMaxScaler(feature_range=(0, 1))
minmax_fit = minmax_scaler.fit(train_data[minmax_cols])
train_data_minmax = minmax_fit.transform(train_data[minmax_cols])
val_data_minmax = minmax_fit.transform(val_data[minmax_cols])
test_data_minmax = minmax_fit.transform(test_data[minmax_cols])
joblib.dump(minmax_fit, 'minmax_scaler.pkl')

# Process Box-Cox and Yeo-Johnson transformations
train_data_bc, val_data_bc, test_data_bc = fit_transform_pipeline(train_data, val_data, test_data, boxcox_cols, method='box-cox')
train_data_yj, val_data_yj, test_data_yj = fit_transform_pipeline(train_data, val_data, test_data, yeojohnson_cols, method='yeo-johnson')

# Combine Transformed Data
train_data_transformed = np.hstack([train_data_minmax, train_data_bc, train_data_yj])
val_data_transformed = np.hstack([val_data_minmax, val_data_bc, val_data_yj])
test_data_transformed = np.hstack([test_data_minmax, test_data_bc, test_data_yj])

# Save Transformed Data
transformed_data_df = pd.DataFrame(np.vstack([train_data_transformed, val_data_transformed, test_data_transformed]),
                                    columns=minmax_cols + boxcox_cols + yeojohnson_cols)
transformed_data_df.to_csv('min_transformed_data.csv', index=False)

# Process Labels Independently
def fit_transform_pipeliney(train_df, val_df, test_df, cols, method='box-cox'):
    pipeline = Pipeline([
        ('power_transformer', PowerTransformer(method=method, standardize=False)),
        ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))
    ])
    pipeline.fit(train_df[cols])
    train_transformed = pipeline.transform(train_df[cols])
    val_transformed = pipeline.transform(val_df[cols])
    test_transformed = pipeline.transform(test_df[cols])
    
    joblib.dump(pipeline, f'{method}_pipeliney.pkl')
    return train_transformed, val_transformed, test_transformed
train_labels_bc, val_labels_bc, test_labels_bc = fit_transform_pipeliney(train_labels, val_labels, test_labels, boxcox_colsy, method='box-cox')

# Combine Transformed Labels
train_labels_transformed = train_labels_bc
val_labels_transformed = val_labels_bc
test_labels_transformed = test_labels_bc
transformed_labels = np.vstack([train_labels_transformed, val_labels_transformed, test_labels_transformed])
