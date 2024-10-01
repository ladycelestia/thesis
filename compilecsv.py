import pandas as pd
from pathlib import Path

# Define path using pathlib for efficiency
path = Path(r'D:\School\ADMU\4Y\SEM 1\MATH 199.11\Final\LMP')

# Define output file
output_file = 'LMP_Complete.csv'

# Initialize a flag to write the header only once
write_header = True

# Function to process each file in chunks and append it to the output CSV
def process_csv_file(file_path, chunk_size=500000):
    global write_header
    for chunk in pd.read_csv(file_path, index_col=None, header=0, engine='python', chunksize=chunk_size, dtype=str):
        # Drop the last row assuming it's the footer
        chunk.drop(chunk.tail(1).index, inplace=True)
        # Convert RUN_TIME column to datetime
        chunk['RUN_TIME'] = pd.to_datetime(chunk['RUN_TIME'], format='mixed')
        # Append the processed chunk to the output CSV
        chunk.to_csv(output_file, mode='a', index=False, header=write_header, chunksize=chunk_size)
        # After writing the first chunk, set write_header to False
        write_header = False

# Efficient reading of multiple CSV files
all_files = path.glob("*.csv")

# Process each CSV file one by one and append results to the output file
for file in all_files:
    process_csv_file(file)

print("Processing complete.")
