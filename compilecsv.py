import pandas as pd
from pathlib import Path

# Define paths using pathlib
data_dirs = {
    "lmp": Path(r'Final\Raw Data\LMP'),
    "hvdc": Path(r'Final\Raw Data\HVDC'),
    "reserve": Path(r'Final\Raw Data\Reserve')
}

# Function to process each category and save as CSV
def process_category(category, path, chunk_size=500000):
    output_csv = f'{category}.csv'
    write_header = True  # Ensure header is written only once

    for file in path.glob("*.csv"):
        for chunk in pd.read_csv(file, index_col=None, header=0, engine='python', chunksize=chunk_size, dtype=str):
            # Drop the last row assuming it's the footer
            chunk.drop(chunk.tail(1).index, inplace=True)
            # Convert RUN_TIME column to datetime
            chunk['RUN_TIME'] = pd.to_datetime(chunk['RUN_TIME'], format='mixed')
            # Append the processed chunk to the output CSV
            chunk.to_csv(output_csv, mode='a', index=False, header=write_header, chunksize=chunk_size)
            # After writing the first chunk, set write_header to False
            write_header = False

# Process each category separately
for category, path in data_dirs.items():
    process_category(category, path)


