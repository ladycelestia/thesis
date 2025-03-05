import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
import pickle

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

def fit_transform_pipeline(train_df, val_df, test_df, cols, method='box-cox'):
    if method:
        pipeline = Pipeline([
            ('power_transformer', PowerTransformer(method=method, standardize=False)),
            ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))
        ])
    else:
        pipeline = Pipeline([
            ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))
        ])
    pipeline.fit(train_df[cols])
    train_transformed = pipeline.transform(train_df[cols])
    val_transformed = pipeline.transform(val_df[cols])
    test_transformed = pipeline.transform(test_df[cols])
    pipeline_file = f'{method}_pipeline.pkl'
    with open(pipeline_file, 'wb') as f:
        pickle.dump(pipeline, f)
    return train_transformed, val_transformed, test_transformed

