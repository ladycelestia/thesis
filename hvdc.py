import pandas as pd

path = r'D:\School\ADMU\4Y\SEM 1\MATH 199.11\Final\HVDC_Complete.csv'  # use your path

df = pd.read_csv(path, index_col=False, header=0, engine='python')

df['RUN_TIME'] = pd.to_datetime(df['RUN_TIME'], format='mixed')

# Categorizes if min, luz, vis
df['MIN'] = df['HVDC_NAME'].apply(lambda x: 'MIN' in x)
df['VIS'] = df['HVDC_NAME'].apply(lambda x: 'VIS' in x)
df['LUZ'] = df['HVDC_NAME'].apply(lambda x: 'LUZ' in x)

# Created flow_vis column to calculate flow_vis
df['FLOW_VIS'] = 0.0  # Initialize as float

# Created separate df for minvis and vis luz
df_minvis = df[df['HVDC_NAME'] == 'MINVIS1'].copy()
df_visluz = df[df['HVDC_NAME'] == 'VISLUZ1'].copy()

# Negative min vis flow from for vis, positive visluz flowfrom for vis
df_minvis.loc[:, 'FLOW_VIS'] = -df_minvis['FLOW_FROM'].astype(float)
df_visluz.loc[:, 'FLOW_VIS'] = df_visluz['FLOW_FROM'].astype(float)

# Merged into the original data frame
df = df.merge(df_minvis[['RUN_TIME', 'FLOW_VIS']], on='RUN_TIME', how='left', suffixes=('', '_minvis'))
df = df.merge(df_visluz[['RUN_TIME', 'FLOW_VIS']], on='RUN_TIME', how='left', suffixes=('', '_visluz'))

# Added the values
df['FLOW_VIS'] = df['FLOW_VIS'].fillna(0) + df['FLOW_VIS_minvis'].fillna(0) + df['FLOW_VIS_visluz'].fillna(0)

# To avoid the double counting, I just made vis values based on mindanao
df['FLOW_VIS'] = df['FLOW_VIS'].where(df['LUZ'], 0)
df['FLOW_MIN'] = df['FLOW_FROM'].where(df['MIN'], 0)
df['FLOW_LUZ'] = df['FLOW_TO'].where(df['LUZ'], 0)

# Grouped
grouped_df = df.groupby('RUN_TIME').agg(
    FLOW_MIN=('FLOW_MIN', 'sum'),
    FLOW_VIS=('FLOW_VIS', 'sum'),
    FLOW_LUZ=('FLOW_LUZ', 'sum')
).reset_index()

# Create a complete time index
min_time = df['RUN_TIME'].min()
max_time = df['RUN_TIME'].max()
complete_time_index = pd.date_range(start=min_time, end=max_time, freq='5min')

# Reindex the grouped DataFrame to include all time periods and fill missing values with 0
grouped_df = grouped_df.set_index('RUN_TIME').reindex(complete_time_index, fill_value=0).reset_index()
grouped_df.columns = ['RUN_TIME', 'FLOW_MIN', 'FLOW_VIS', 'FLOW_LUZ']

# Write the result to the output file
grouped_df.to_csv('tryhvdc_processed.csv', index=False)

print("Processing complete.")