import pandas as pd
import numpy as np

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
        df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
        
        df.drop(columns=['YEAR', 'MONTH', 'DAY'], inplace=True)
        df = df[(df['DATE'] >= '2022-01-01') & (df['DATE'] <= '2023-12-31')]
        df.rename(columns={
            "RAINFALL": f"RAINFALL_{region}",
            "TMAX": f"TMAX_{region}",
            "TMIN": f"TMIN_{region}"
        }, inplace=True)
        
        if merged_df is None:
            merged_df = df  
        else:
            merged_df = pd.merge(merged_df, df, on="DATE", how="outer")  # Outer join to keep all dates

    # Sort by DATE
    merged_df.sort_values(by="DATE", inplace=True)

    # Ensure DATE is the first column
    cols = ['DATE'] + [col for col in merged_df.columns if col != 'DATE']
    merged_df = merged_df[cols]
    merged_df = clean_columns(merged_df, substrings_ffill=['RAINFALL'], substrings_interpolate=['TMAX', 'TMIN'])
    return merged_df

base_path = "Final/Raw Data/S-092024-046 - Mercado/"
region_groups = {
    "Luzon": [
        "Cubi Point", "NAIA", "Science Garden", "San Jose", "Tayabas",
        "CLSU", "Tanay", "Ambulong", "Casiguran", "Clark", "Calapan"
    ],
    "Visayas": [
        "Catbalogan", "Roxas City", "Catarman", "Maasin", "Dumaguete"
    ],
    "Mindanao": [
        "Davao City", "Surigao", "Zamboanga", "Dipolog", "Butuan",
        "Malaybalay", "General Santos", "Cotabato"
    ]
}


weatherluzon = process_weather_data(base_path, region_groups["Luzon"])
weathervisayas = process_weather_data(base_path, region_groups["Visayas"])
weathermindanao =  process_weather_data(base_path, region_groups["Mindanao"])

weatherluzon.to_csv("LuzWeather.csv", index=False)
weathervisayas.to_csv("VisWeather.csv", index=False)
weathermindanao.to_csv("MinWeather.csv", index=False)
