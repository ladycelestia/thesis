import pandas as pd
from functools import reduce
# File paths
lmp_file = r'Final\Final Codes\Final Data\lmp.csv'
reserve_file = r'Final\Final Codes\Final Data\reserve.csv'
output_file = 'Final_Daily_Results.csv'

# Parameters
chunk_size = 50000
date_start = pd.Timestamp('2022-01-01')
date_end = pd.Timestamp('2023-12-31')


# Unified function for GWAP, LWAP, and Reserve GWAPs
def price_weighted_average(input_file, resource_filter, price_column, output_column, commodity_filter=None):
    usecols = ['RUN_TIME', 'RESOURCE_TYPE', 'REGION_NAME', price_column, 'SCHED_MW']
    if commodity_filter:
        usecols.append('COMMODITY_TYPE')
        
    chunks = []

    for chunk in pd.read_csv(input_file, usecols=usecols, chunksize=chunk_size, parse_dates=['RUN_TIME']):
        # Apply filters
        filtered = chunk[chunk['RESOURCE_TYPE'] == resource_filter].copy(deep=True)
        if commodity_filter:
            filtered = filtered[filtered['COMMODITY_TYPE'] == commodity_filter]

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

    # Calculate weighted average and handle division by zero
    final[output_column] = final['PRICE_x_SCHED'] / final['SCHED_MW']
    final.loc[final['SCHED_MW'] == 0, output_column] = 0

    return final[['REGION_NAME', 'RUN_TIME', output_column]]


# Function to merge multiple DataFrames
def merge_results(dataframes):
    return reduce(lambda left, right: pd.merge(left, right, on=['REGION_NAME', 'RUN_TIME'], how='outer'), dataframes)


# Calculate all required values using the single function
results = [
    price_weighted_average(lmp_file, 'G', 'LMP', 'GWAP'),
    price_weighted_average(lmp_file, 'NL', 'LMP', 'LWAP'),
    price_weighted_average(reserve_file, 'G', 'PRICE', 'Reserve_GWAP_Fr', commodity_filter='Fr'),
    price_weighted_average(reserve_file, 'G', 'PRICE', 'Reserve_GWAP_Ru', commodity_filter='Ru'),
    price_weighted_average(reserve_file, 'G', 'PRICE', 'Reserve_GWAP_Rd', commodity_filter='Rd'),
    price_weighted_average(reserve_file, 'G', 'PRICE', 'Reserve_GWAP_Dr', commodity_filter='Dr')
]

# Merge all DataFrames at once
prices = merge_results(results)

# Fill NaNs with 0 for missing commodities
prices.fillna(0, inplace=True)
# Filter by date range before exporting
prices = prices[
    (prices['RUN_TIME'] >= date_start) & (prices['RUN_TIME'] <= date_end)
]

# Export to CSV
prices.to_csv(output_file, index=False)

