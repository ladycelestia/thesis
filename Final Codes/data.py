import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from typing import List, Dict, Union, Optional, Tuple
from functools import reduce


class DataProcessor:
    def __init__(self, base_path):
        
        self.base_path = base_path
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

    def price_weighted_average(self, input_file, resource_filter, price_column, output_column, date_start, date_end, region_filter, commodity_filter):
        
        usecols = ['RUN_TIME', 'RESOURCE_TYPE', 'REGION_NAME', price_column, 'SCHED_MW']
        if commodity_filter:
            usecols.append('COMMODITY_TYPE')

        chunks = []

        for chunk in pd.read_csv(input_file, usecols=usecols, chunksize=500000, parse_dates=['RUN_TIME']):
            # Apply filters
            filtered = chunk[chunk['RESOURCE_TYPE'] == resource_filter].copy(deep=True)
            if commodity_filter:
                filtered = filtered[filtered['COMMODITY_TYPE'] == commodity_filter]
            if region_filter:
                filtered = filtered[filtered['REGION_NAME'] == region_filter]

            # Set negative prices to 0
            filtered.loc[filtered[price_column] < 0, price_column] = 0

            # Calculate weighted contribution
            filtered.loc[:, 'PRICE_x_SCHED'] = filtered[price_column] * filtered['SCHED_MW']

            # Aggregate sums per group within this chunk
            grouped = filtered.groupby(['REGION_NAME', pd.Grouper(key='RUN_TIME', freq='D')]).agg({
                'PRICE_x_SCHED': 'sum',
                'SCHED_MW': 'sum'
            }).reset_index()

            chunks.append(grouped)

        # Combine all aggregated results
        combined = pd.concat(chunks, ignore_index=True)

        # Final aggregation across all chunks
        final = combined.groupby(['REGION_NAME', 'RUN_TIME']).agg({
            'PRICE_x_SCHED': 'sum',
            'SCHED_MW': 'sum'
        }).reset_index()

        # Ensure all expected dates are included
        if date_start and date_end:
            all_dates = pd.date_range(start=date_start, end=date_end, freq='D')
            all_dates_df = pd.DataFrame({'RUN_TIME': all_dates})

            final = final.merge(all_dates_df, on='RUN_TIME', how='right')

        # Calculate weighted average and handle division by zero
        final[output_column] = final['PRICE_x_SCHED'] / final['SCHED_MW']
        final.loc[final['SCHED_MW'] == 0, output_column] = 0
        final.fillna(0, inplace=True)

        return final[['RUN_TIME', output_column]]

    def process_hvdc_data(self, input_path, region_filter) :
        
        df = pd.read_csv(input_path)
        df['RUN_TIME'] = pd.to_datetime(df['RUN_TIME'], format='mixed')

        # Calculate flow values
        df['FLOW_MIN'] = df.apply(lambda x: x['FLOW_FROM'] if 'MIN' in x['HVDC_NAME'] else 0, axis=1)
        df['FLOW_LUZ'] = df.apply(lambda x: x['FLOW_TO'] if 'LUZ' in x['HVDC_NAME'] else 0, axis=1)

        # Calculate FLOW_VIS from MINVIS1 and VISLUZ1
        df['FLOW_VIS'] = df.apply(
            lambda x: -x['FLOW_FROM'] if x['HVDC_NAME'] == 'MINVIS1' else (x['FLOW_FROM'] if x['HVDC_NAME'] == 'VISLUZ1' else 0),
            axis=1
        )

        # Group and aggregate daily
        grouped_df = df.groupby(pd.Grouper(key='RUN_TIME', freq='D')).agg(
            FLOW_MIN=('FLOW_MIN', 'sum'),
            FLOW_VIS=('FLOW_VIS', 'sum'),
            FLOW_LUZ=('FLOW_LUZ', 'sum')
        ).reset_index()

        # Export the result
        return grouped_df[['RUN_TIME', f'FLOW_{region_filter}']]

    def clean_columns(
        self, 
        df, 
        substrings_ffill, 
        substrings_interpolate
    ):
        df = df.copy()  # Avoid modifying original

        # Replace -999 and -1 with NaN (for processing)
        df.replace(-1, 0, inplace=True)
        df.replace(-999, np.nan, inplace=True)

        # Forward fill for specified columns
        for col in df.filter(regex='|'.join(substrings_ffill), axis=1).columns:
            df[col] = df[col].ffill().fillna(0)  # Fill remaining NaN with 0

        # Interpolation for specified columns
        for col in df.filter(regex='|'.join(substrings_interpolate), axis=1).columns:
            df[col] = df[col].interpolate(method='linear').fillna(0)  # Fill remaining NaN with 0

        return df

    def process_weather_data(self, path, regions):
        
        merged_df = None  
        for region in regions:
            file_path = f"{path}{region} Daily Data.csv"

            df = pd.read_csv(file_path)
            df['RUN_TIME'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
            
            df.drop(columns=['YEAR', 'MONTH', 'DAY'], inplace=True)
           
            df.rename(columns={
                "RAINFALL": f"RAINFALL_{region}",
                "TMAX": f"TMAX_{region}",
                "TMIN": f"TMIN_{region}"
            }, inplace=True)
            
            if merged_df is None:
                merged_df = df  
            else:
                merged_df = pd.merge(merged_df, df, on="RUN_TIME", how="outer")  

        merged_df.sort_values(by="RUN_TIME", inplace=True)

        cols = ['RUN_TIME'] + [col for col in merged_df.columns if col != 'RUN_TIME']
        merged_df = merged_df[cols]
        merged_df = self.clean_columns(
            merged_df, 
            substrings_ffill=['RAINFALL'], 
            substrings_interpolate=['TMAX', 'TMIN']
        )
        return merged_df

    @staticmethod
    def merge_results(dataframes):

        return reduce(lambda left, right: pd.merge(left, right, on=['RUN_TIME'], how='outer'), dataframes)

    def demand(self, path, region_name):
        file_path = f"{path}{region_name}HourlyDemand.csv"
        return pd.read_csv(file_path, parse_dates=["RUN_TIME"])

    def process_region(self, lmp_file, hvdc_file, reserve_file, demand_file, weather_path, region_name, weather_regions, date_start, date_end):
        
        pickle_file = os.path.join(self.script_dir, "Final Data", f"{region_name}_Daily_Complete.pkl")
        
        if os.path.exists(pickle_file):
            with open(pickle_file, "rb") as f:
                return pickle.load(f)
        
        df_list = [
            self.price_weighted_average(lmp_file, 'G', 'LMP', 'GWAP', date_start, date_end, f'C{region_name}'),
            self.price_weighted_average(lmp_file, 'NL', 'LMP', 'LWAP', date_start, date_end, f'C{region_name}'),
            self.process_hvdc_data(hvdc_file, region_name),
            self.price_weighted_average(reserve_file, 'G', 'PRICE', 'Reserve_GWAP_Fr', date_start, date_end, f'C{region_name}', 'Fr'),
            self.price_weighted_average(reserve_file, 'G', 'PRICE', 'Reserve_GWAP_Ru', date_start, date_end, f'C{region_name}', 'Ru'),
            self.price_weighted_average(reserve_file, 'G', 'PRICE', 'Reserve_GWAP_Rd', date_start, date_end, f'C{region_name}', 'Rd'),
            self.price_weighted_average(reserve_file, 'G', 'PRICE', 'Reserve_GWAP_Dr', date_start, date_end, f'C{region_name}', 'Dr'),
            self.demand(demand_file, region_name),
            self.process_weather_data(weather_path, weather_regions)
        ]
        
        final = self.merge_results(df_list)
        final = final[
            (final['RUN_TIME'] >= date_start) & (final['RUN_TIME'] <= date_end)
        ]
        final.reset_index(drop=True, inplace=True)
        final.set_index('RUN_TIME', inplace=True)
        
        with open(pickle_file, "wb") as f:
            pickle.dump(final, f)
        
        return final

    def load_data(self, region_name, transformed):
        
        folder = "Final Data"
        filename = f"{region_name}_Transformed_Daily_Complete.pkl" if transformed else f"{region_name}_Daily_Complete.pkl"
        pickle_file = os.path.join(self.script_dir, folder, filename)
        
        if os.path.exists(pickle_file):
            with open(pickle_file, "rb") as f:
                return pickle.load(f)
        else:
            print("Data not found")
            return None

    @staticmethod
    def split_data(X, use_val):
        
        if use_val:
            train_size = int(0.6 * len(X))
            val_size = int(0.2 * len(X))

            train = X[:train_size]
            val = X[train_size:train_size + val_size]
            test = X[train_size + val_size:]

            return train, val, test
        
        else:
            train_size = int(0.8 * len(X))

            train = X[:train_size]
            test = X[train_size:]

            return train, test
        
class DataTransformer:
    
    def __init__(self, region_name):
       
        self.region_name = region_name
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Final Data")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Store classification and transformation results
        self.minmax_cols: List[str] = []
        self.boxcox_cols: List[str] = []
        self.yeojohnson_cols: List[str] = []
        
        # Store pipelines
        self.pipelines: Dict[str, Pipeline] = {}
    
    def _get_cache_path(self, method, is_pipeline):
        
        filename = (
            f"{self.region_name}__{method}_pipeline.pkl" if is_pipeline
            else f"{self.region_name}_Transformed_Daily_Complete.pkl"
        )
        return os.path.join(self.cache_dir, filename)
    
    def classify_features(self, data):
        
        self.minmax_cols.clear()
        self.boxcox_cols.clear()
        self.yeojohnson_cols.clear()
        
        for column in data.columns:
            skewness = data[column].skew()
            kurtosis = data[column].kurtosis()
            is_positive = np.all(data[column] > 0)

            if -1 <= skewness <= 1 and -1 <= kurtosis <= 1:
                self.minmax_cols.append(column)
            elif is_positive:
                self.boxcox_cols.append(column)
            else:
                self.yeojohnson_cols.append(column)
    
    def _create_pipeline(self, method):
        
        if method in ["box-cox", "yeo-johnson"]:
            return Pipeline([
                ('power_transformer', PowerTransformer(method=method, standardize=False)),
                ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))
            ])
        elif method == "minmax":
            return Pipeline([
                ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))
            ])
        else:
            raise ValueError(f"Unsupported transformation method: {method}")
    
    def fit_pipeline(self, df, cols, method):
        
        cache_path = self._get_cache_path(method,is_pipeline=True)
        
        # Check cache
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        # Create and fit pipeline
        pipeline = self._create_pipeline(method)
        pipeline.fit(df[cols])
        
        # Cache pipeline
        with open(cache_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        return pipeline
    
    def transform_data(
        self, 
        data, 
        use_val
    ):
        
        cache_path = self._get_cache_path(method='transformed', is_pipeline=False)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        # Classify features
        self.classify_features(data)
        
        # Split data
        from datascript import split_data  # Assuming this is your splitting function
        train, val, test = (split_data(data, True) if use_val 
                            else (split_data(data, False)[0], None, split_data(data, False)[1]))
        
        # Fit pipelines
        self.pipelines = {
            "boxcox": self.fit_pipeline(train, self.boxcox_cols, method='box-cox'),
            "yeojohnson": self.fit_pipeline(train, self.yeojohnson_cols, method='yeo-johnson'),
            "minmax": self.fit_pipeline(train, self.minmax_cols, method='minmax')
        }
        
        def apply_transform(df: pd.DataFrame) -> pd.DataFrame:
            """Apply transformations to a DataFrame."""
            transformed_parts = {
                method: pd.DataFrame(
                    pipeline.transform(df[cols]), 
                    columns=cols, 
                    index=df.index
                )
                for method, pipeline, cols in [
                    ("minmax", self.pipelines["minmax"], self.minmax_cols),
                    ("boxcox", self.pipelines["boxcox"], self.boxcox_cols),
                    ("yeojohnson", self.pipelines["yeojohnson"], self.yeojohnson_cols)
                ]
            }
            return pd.concat(transformed_parts.values(), axis=1)[data.columns]
        
        # Transform datasets
        train_transformed = apply_transform(train)
        test_transformed = apply_transform(test)
        
        # Combine transformed data
        if use_val:
            val_transformed = apply_transform(val)
            full_transformed = pd.concat([train_transformed, val_transformed, test_transformed])
        else:
            full_transformed = pd.concat([train_transformed, test_transformed])
        
        # Cache transformed data
        with open(cache_path, "wb") as f:
            pickle.dump(full_transformed, f)
        
        return full_transformed

