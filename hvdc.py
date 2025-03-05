import pandas as pd

def process_hvdc_data(input_path, output_path, date_start, date_end):
    # Load data
    df = pd.read_csv(input_path)
    df['RUN_TIME'] = pd.to_datetime(df['RUN_TIME'], format='mixed')

    # Filter by date range
    df = df[(df['RUN_TIME'] >= date_start) & (df['RUN_TIME'] <= date_end)]

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
    grouped_df.to_csv(output_path, index=False)

# Usage
input_file = r'Final\Final Codes\Final Data\hvdc.csv'
output_file = 'tryhvdc_processed.csv'
date_start = pd.Timestamp('2022-01-01')
date_end = pd.Timestamp('2023-12-31')

process_hvdc_data(input_file, output_file, date_start, date_end)
