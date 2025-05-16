import os
import pandas as pd
from glob import glob
from datetime import datetime, timedelta

base_dir = "/mnt/hda/realtime/Dataset"
output_dir = "/mnt/hda/realtime/neurips"
from tqdm import tqdm
os.makedirs(output_dir, exist_ok=True)
def fahrenheit_to_kelvin(fahrenheit):
    return (5/9) * (fahrenheit - 32) + 273.15
variables = {
    "MXSPD": 999.9,
    "DEWP": 9999.9,
    "SLP": 9999.9,  
    "WDSP": 999.9,
    "MAX": 9999.9, 
    "MIN": 9999.9, 
}

for year in range(2017, 2025):
    year_path = os.path.join(base_dir, str(year))
    if not os.path.exists(year_path):
        continue  

    csv_files = glob(os.path.join(year_path, "*.csv"))


    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    while start_date <= end_date:
        date_str = start_date.strftime("%Y-%m-%d")  
        daily_data = {var: [] for var in variables} 

        for file in tqdm(csv_files):
            df = pd.read_csv(file, dtype={'STATION': str})

            df = df[df['DATE'] == date_str]

            for var, missing_value in variables.items():
                if var in df.columns:

                    df_var = df[df[var] != missing_value][['LONGITUDE', 'LATITUDE', 'STATION', var]]

                    daily_data[var].extend(df_var.values.tolist())


        for var, records in daily_data.items():
            if not records: 
                continue

            var_dir = os.path.join(output_dir, var)
            os.makedirs(var_dir, exist_ok=True)
            file_path = os.path.join(var_dir, f"{date_str}.csv")

            df_out = pd.DataFrame(records, columns=["longitude", "latitude", "station_id", "observation_value"])
            df_out = df_out.dropna(how='any')
            if var in ["MAX","MIN","TEMP","DEWP"]:
                df_out['observation_value'] = df_out['observation_value'].apply(fahrenheit_to_kelvin)
            df_out = df_out.drop_duplicates(subset=['station_id'], keep='first')
            df_out.to_csv(file_path, index=False)

        print(f"✅ {date_str} finish")
        
        start_date += timedelta(days=1)
