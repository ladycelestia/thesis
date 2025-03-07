import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
import pickle
from datascript import split_data
import os
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def fit_pipeline(df, cols, regionname,target_label,method='box-cox'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if len(cols)==1:
        pickle_file = os.path.join(script_dir, "Final Data", f"{regionname}_target_{target_label}_{method}_pipeline.pkl")
    else:
        pickle_file = os.path.join(script_dir, "Final Data", f"{regionname}_features_{target_label}_{method}_pipeline.pkl")

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

def transform_data(data, regionname, target_label, use_val=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    feature_pickle_file = os.path.join(script_dir, "Final Data", f"{regionname}_{target_label}_Transformed_Features.pkl")
    target_pickle_file = os.path.join(script_dir, "Final Data", f"{regionname}_{target_label}_Transformed_Target.pkl")

    if os.path.exists(feature_pickle_file) and os.path.exists(target_pickle_file):
        with open(feature_pickle_file, "rb") as f:
            transformed_features = pickle.load(f)
        with open(target_pickle_file, "rb") as f:
            transformed_labels = pickle.load(f)
        return transformed_features, transformed_labels

    # Ensure target_label is in the dataset
    if target_label not in data.columns:
        raise ValueError(f"Target label '{target_label}' not found in dataset.")

    # Remove the non-target label
    features = data.drop(columns=[col for col in ["GWAP", "LWAP"] if col != target_label])

    # Classify features (including the target label)
    minmax_cols, boxcox_cols, yeojohnson_cols = classify_features(features)

    # Split data
    if use_val:
        train, val, test = split_data(features, True)
        train_labels, val_labels, test_labels = train[[target_label]], val[[target_label]], test[[target_label]]
    else:
        train, test = split_data(features, False)
        train_labels, test_labels = train[[target_label]], test[[target_label]]

    # Fit transformation pipelines
    pipelines = {
        "boxcox": fit_pipeline(train[boxcox_cols], boxcox_cols, regionname, target_label,method='box-cox'),
        "yeojohnson": fit_pipeline(train[yeojohnson_cols], yeojohnson_cols, regionname, target_label,method='yeo-johnson'),
        "minmax": fit_pipeline(train[minmax_cols], minmax_cols, regionname, target_label,method='minmax')
    }

    # Determine the transformation method for the target label

    label_pipeline = fit_pipeline(train_labels, [target_label], regionname,target_label, method='box-cox')


    def apply_transform(df):
        transformed_parts = {
            "minmax": pd.DataFrame(pipelines["minmax"].transform(df[minmax_cols]), 
                                   columns=minmax_cols, index=df.index),
            "boxcox": pd.DataFrame(pipelines["boxcox"].transform(df[boxcox_cols]), 
                                   columns=boxcox_cols, index=df.index),
            "yeojohnson": pd.DataFrame(pipelines["yeojohnson"].transform(df[yeojohnson_cols]), 
                                       columns=yeojohnson_cols, index=df.index)
        }
        return pd.concat(transformed_parts.values(), axis=1)[df.columns]

    # Transform features
    train_transformed = apply_transform(train)
    test_transformed = apply_transform(test)

    if use_val:
        val_transformed = apply_transform(val)
        transformed_features = pd.concat([train_transformed, val_transformed, test_transformed])
    else:
        transformed_features = pd.concat([train_transformed, test_transformed])

    # Transform labels separately
    train_labels_transformed = label_pipeline.transform(train_labels)
    test_labels_transformed = label_pipeline.transform(test_labels)

    if use_val:
        val_labels_transformed = label_pipeline.transform(val_labels)
        transformed_labels = pd.concat([
            pd.DataFrame(train_labels_transformed, index=train_labels.index, columns=[target_label]),
            pd.DataFrame(val_labels_transformed, index=val_labels.index, columns=[target_label]),
            pd.DataFrame(test_labels_transformed, index=test_labels.index, columns=[target_label])
        ])
    else:
        transformed_labels = pd.concat([
            pd.DataFrame(train_labels_transformed, index=train_labels.index, columns=[target_label]),
            pd.DataFrame(test_labels_transformed, index=test_labels.index, columns=[target_label])
        ])

    # Save transformed data
    with open(feature_pickle_file, "wb") as f:
        pickle.dump(transformed_features, f)
    with open(target_pickle_file, "wb") as f:
        pickle.dump(transformed_labels, f)

    return transformed_features, transformed_labels



def inverse_transform_data(data, regionname, target_label):
    script_dir = os.path.dirname(os.path.abspath(__file__))
   
    # Check if data is numpy array or tensor and convert to DataFrame
    original_type = None
    original_shape = None
    
    if isinstance(data, np.ndarray):
        original_type = "numpy"
        original_shape = data.shape

        data = pd.DataFrame(data.reshape(-1, 1), columns=[target_label])
        

    elif isinstance(data, torch.Tensor):
        original_type = "tensor"
        data_np = data.cpu().numpy() if data.is_cuda else data.numpy()
        original_shape = data_np.shape
        
        data = pd.DataFrame(data_np.reshape(-1, 1), columns=[target_label])


    
    # Now data is guaranteed to be a DataFrame
    if(len(data.columns)==1):
        pipeline_files = {
            "boxcox": os.path.join(script_dir, "Final Data", f"{regionname}_target_{target_label}_box-cox_pipeline.pkl"),
        }
    else:
        pipeline_files = {
            "boxcox": os.path.join(script_dir, "Final Data", f"{regionname}_features_{target_label}_box-cox_pipeline.pkl"),
            "yeojohnson": os.path.join(script_dir, "Final Data", f"{regionname}_features_{target_label}_yeo-johnson_pipeline.pkl"),
            "minmax": os.path.join(script_dir, "Final Data", f"{regionname}_features_{target_label}_minmax_pipeline.pkl")
        }

    pipelines = {}
    for method, file in pipeline_files.items():
        if os.path.exists(file):
            with open(file, "rb") as f:
                pipelines[method] = pickle.load(f)
        else:
            raise FileNotFoundError(f"Pipeline file {file} not found. Ensure transformations were applied first.")
    
    boxcox_cols = pipelines["boxcox"].named_steps["power_transformer"].feature_names_in_ if "boxcox" in pipelines else []
    yeojohnson_cols = pipelines["yeojohnson"].named_steps["power_transformer"].feature_names_in_ if "yeojohnson" in pipelines else []
    minmax_cols = pipelines["minmax"].named_steps["minmax_scaler"].feature_names_in_ if "minmax" in pipelines else []
    
    def apply_inverse_transform(df):
        inverse_parts = {}

        if "boxcox" in pipelines:
            inverse_parts["boxcox"] = pd.DataFrame(
                pipelines["boxcox"].inverse_transform(df[boxcox_cols]), 
                columns=boxcox_cols, index=df.index
            )
        if "yeojohnson" in pipelines:
            inverse_parts["yeojohnson"] = pd.DataFrame(
                pipelines["yeojohnson"].inverse_transform(df[yeojohnson_cols]), 
                columns=yeojohnson_cols, index=df.index
            )
        if "minmax" in pipelines:
            inverse_parts["minmax"] = pd.DataFrame(
                pipelines["minmax"].inverse_transform(df[minmax_cols]), 
                columns=minmax_cols, index=df.index
            )

        return pd.concat(inverse_parts.values(), axis=1)[df.columns] if inverse_parts else df

    result_df = apply_inverse_transform(data)
    
    # Return the result in the same format as the input
    if original_type == "numpy":
        result_np = result_df.values
        # Reshape back to original shape if needed (handles 1D case)
        if len(original_shape) == 1:
            result_np = result_np.squeeze()
        return result_np
    elif original_type == "tensor":
        result_np = result_df.values
        if len(original_shape) == 1:
            result_np = result_np.squeeze()
        return torch.tensor(result_np)
    else:
        return result_df


