import pandas as pd
import numpy as np
from functools import reduce
import pickle
import os
import pandas as pd

def price_weighted_average(input_file, resource_filter, price_column, output_column, date_start=None, date_end=None, region_filter=None, commodity_filter=None, ):
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

def process_hvdc_data(input_path, region_filter=None):
    
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

def clean_columns(df, substrings_ffill, substrings_interpolate):
    df = df.copy()  # Avoid modifying original

    # Replace -999 and -1 with NaN (for processing)
    df.replace(-1, 0, inplace=True)

    # Replace -999 with NaN for processing
    df.replace(-999, np.nan, inplace=True)

    # Forward fill for rainfall columns
    for col in df.filter(regex='|'.join(substrings_ffill), axis=1).columns:
        df[col] = df[col].ffill().fillna(0)  # Fill remaining NaN with 0

    # Interpolation for temperature columns
    for col in df.filter(regex='|'.join(substrings_interpolate), axis=1).columns:
        df[col] = df[col].interpolate(method='linear').fillna(0)  # Fill remaining NaN with 0

    return df

def process_weather_data(base_path, regions):
    merged_df = None  
    for region in regions:
        file_path = f"{base_path}{region} Daily Data.csv"

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
    merged_df = clean_columns(merged_df, substrings_ffill=['RAINFALL'], substrings_interpolate=['TMAX', 'TMIN'])
    return merged_df

def merge_results(dataframes):
    return reduce(lambda left, right: pd.merge(left, right, on=['RUN_TIME'], how='outer'), dataframes)

def process_region(lmp_file,hvdc_file,reserve_file,weatherpath,regionname,weatherregion,date_start,date_end):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(script_dir, f"{regionname}_Daily_Complete.pkl")
    
    if os.path.exists(pickle_file):
            with open(pickle_file, "rb") as f:
                df = pickle.load(f)
            return df
    else:
        df= [
            price_weighted_average(lmp_file, 'G', 'LMP', 'GWAP', date_start,date_end,f'C{regionname}'),
            price_weighted_average(lmp_file, 'NL', 'LMP', 'LWAP', date_start,date_end,f'C{regionname}'),
            process_hvdc_data(hvdc_file, regionname),
            price_weighted_average(reserve_file, 'G', 'PRICE', 'Reserve_GWAP_Fr',date_start,date_end, f'C{regionname}', 'Fr'),
            price_weighted_average(reserve_file, 'G', 'PRICE', 'Reserve_GWAP_Ru',date_start,date_end, f'C{regionname}', 'Ru'),
            price_weighted_average(reserve_file, 'G', 'PRICE', 'Reserve_GWAP_Rd',date_start,date_end, f'C{regionname}', 'Rd'),
            price_weighted_average(reserve_file, 'G', 'PRICE', 'Reserve_GWAP_Dr',date_start,date_end, f'C{regionname}', 'Dr'),
            process_weather_data(weatherpath, weatherregion)
        ]
        final = merge_results(df)
        final = final[
        (final['RUN_TIME'] >= date_start) & (final['RUN_TIME'] <= date_end)
        ]
        final.reset_index(drop=True, inplace=True)
        final.set_index('RUN_TIME', inplace=True)
        with open(pickle_file, "wb") as f:
            pickle.dump(final, f)
        return final
    

def load_data(regionname):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_file = os.path.join(script_dir, f"{regionname}_Daily_Complete.pkl")
    if os.path.exists(pickle_file):
            with open(pickle_file, "rb") as f:
                df = pickle.load(f)
            return df
    else:
        print(f"Pickle file for {regionname} not found.")
        return None
    
def split_data(X, use_val=True):
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


