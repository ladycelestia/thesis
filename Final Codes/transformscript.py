import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
import pickle
from datascript import split_data
import os

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

def fit_pipeline(df, cols, regionname,method='box-cox'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(script_dir, "Final Data", f"{regionname}__{method}_pipeline.pkl")
    
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
    
    if method == "box-cox" or method == "yeo-johnson":
        pipeline = Pipeline([
            ('power_transformer', PowerTransformer(method=method, standardize=False)),
            ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))
        ])
    elif method == "minmax":
        pipeline = Pipeline([
            ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))
        ])

    pipeline.fit(df[cols])
        
    with open(pickle_file, 'wb') as f:
        pickle.dump(pipeline, f)
    return pipeline

def transform_data(data, regionname, use_val=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(script_dir, "Final Data", f"{regionname}_Transformed_Daily_Complete.pkl")

    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            df = pickle.load(f)
        return df

    # Otherwise, process the data
    minmax_cols, boxcox_cols, yeojohnson_cols = classify_features(data)

    # Split data accordingly
    if use_val:
        train, val, test = split_data(data, True)
    else:
        train, test = split_data(data, False)  # Assume split_data handles this case

    pipelines = {
        "boxcox": fit_pipeline(train, boxcox_cols, regionname, method='box-cox'),
        "yeojohnson": fit_pipeline(train, yeojohnson_cols, regionname, method='yeo-johnson'),
        "minmax": fit_pipeline(train, minmax_cols, regionname, method='minmax')
    }

    def apply_transform(df):
        transformed_parts = {
            "minmax": pd.DataFrame(pipelines["minmax"].transform(df[minmax_cols]), 
                                   columns=minmax_cols, index=df.index),
            "boxcox": pd.DataFrame(pipelines["boxcox"].transform(df[boxcox_cols]), 
                                   columns=boxcox_cols, index=df.index),
            "yeojohnson": pd.DataFrame(pipelines["yeojohnson"].transform(df[yeojohnson_cols]), 
                                       columns=yeojohnson_cols, index=df.index)
        }
        return pd.concat(transformed_parts.values(), axis=1)[data.columns]

    # Transform each dataset
    train_transformed = apply_transform(train)
    test_transformed = apply_transform(test)
    
    if use_val:
        val_transformed = apply_transform(val)
        full_transformed = pd.concat([train_transformed, val_transformed, test_transformed])
    else:
        full_transformed = pd.concat([train_transformed, test_transformed])

    with open(pickle_file, "wb") as f:
        pickle.dump(full_transformed, f)

    return full_transformed


