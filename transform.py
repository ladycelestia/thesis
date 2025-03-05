import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
import pickle
import os
from finaldata import split_data
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

    pipeline_file = f'{regionname}_{method}_pipeline.pkl'
    if os.path.exists(pipeline_file):
        with open(pipeline_file, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
    else:
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
        
        with open(pipeline_file, 'wb') as f:
            pickle.dump(pipeline, f)
        return pipeline



def transform_data( data, regionname):
    minmax_cols, boxcox_cols, yeojohnson_cols = classify_features(data)

    # Fit and get pipeline instances
    

    train, val, test = split_data(data,True)


    boxcox_pipeline = fit_pipeline(train, boxcox_cols, regionname, method='box-cox')
    yeojohnson_pipeline = fit_pipeline(train, yeojohnson_cols, regionname, method='yeo-johnson')
    minmax_pipeline = fit_pipeline(train, minmax_cols, regionname,method='minmax')

    # Ensure all datasets have the same columns
    cols = train.columns  # Store feature names from training set
    
    train_transformed = boxcox_pipeline.transform(train[cols])
    val_transformed = yeojohnson_pipeline.transform(val[cols]) if val is not None else None
    test_transformed = minmax_pipeline.transform(test[cols])

    return train_transformed, val_transformed, test_transformed

