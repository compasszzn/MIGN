import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Define the folder path
folder_path = Path('/mnt/hda/zzn/realtime/nips')

# Get all CSV files in the folder
# csv_files = sorted(folder_path.glob('*.csv'), key=lambda f: f.stat().st_size)



# Initialize variable to store the union of primary_station_id
variables = [
    "DEWP",
    # "MXSPD",
    # "GUST",
    # "STP",
# "PRCP",
# "SLP",
# "WDSP",
#  "MAX",
#  "MIN"
]
for variable in variables:
    station_info = pd.DataFrame(columns=['station_id', 'latitude', 'longitude'])
    csv_files = sorted([
            f for f in (folder_path/variable).glob('*.csv') 
            # if f.stem[:4] in ["2022","2023","2024"]
            ])
    for file in tqdm(csv_files):
        df = pd.read_csv(file, usecols=['station_id','latitude','longitude'], dtype={'station_id': str})
        station_info = pd.concat([station_info, df])
        station_info = station_info.drop_duplicates(subset=['station_id'], keep='first')

    # Save the DataFrame to a CSV file
    output_csv = f"{folder_path}/unique_primary_station_ids_{variable}_tune.csv"
    station_info.to_csv(output_csv, index=False)

    print(f"Union of primary_station_id values saved to {output_csv}")
