import pandas as pd

input_file = r'D:\School\ADMU\4Y\SEM 1\MATH 199.11\Final\LMP_Complete.csv'
output_file = 'LWAP.csv'

# Define chunk size (adjust based on your system's capacity)
chunk_size = 50000

# Columns to retain (drop unnecessary columns during reading)
usecols = ['RUN_TIME', 'RESOURCE_TYPE', 'REGION_NAME', 'LMP', 'SCHED_MW']

# Initialize an empty list to hold processed chunks
chunks = []

# Read and process file in chunks
for chunk in pd.read_csv(input_file, usecols=usecols, chunksize=chunk_size, parse_dates=['RUN_TIME']):
    # Filter rows where RESOURCE_TYPE is 'G'
    chunk = chunk[chunk['RESOURCE_TYPE'] == 'G']
    
    # Set negative LMP prices to 0
    chunk.loc[chunk['LMP'] < 0, 'LMP'] = 0
    
    # Group by REGION_NAME and RUN_TIME with 5-minute frequency
    grouped = chunk.groupby(['REGION_NAME', pd.Grouper(key='RUN_TIME', freq='5min')])

    # Calculate the weighted average (GWAP) for each group
    def calculate_weighted_avg(x):
        total_sched_mw = x['SCHED_MW'].sum()
        if total_sched_mw == 0:
            return pd.Series({'GWAP': 0})  # Handle division by zero
        return pd.Series({
            'GWAP': (x['LMP'] * x['SCHED_MW']).sum() / total_sched_mw
        })

    # Apply the function to the group and exclude the grouping columns from the operation
    weighted_avg = grouped.apply(calculate_weighted_avg, include_groups=False).reset_index()

    # Append processed chunk to the list
    chunks.append(weighted_avg)

# Concatenate all processed chunks into one DataFrame
result = pd.concat(chunks, ignore_index=True)

# Write the result to the output file in chunks
result.to_csv(output_file, index=False, chunksize=50000)

print("Processing complete.")