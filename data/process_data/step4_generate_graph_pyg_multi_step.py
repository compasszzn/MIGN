import pandas as pd
import numpy as np
import argparse
# Define the folder path
from pathlib import Path
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import os

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    r = 6371000 / 1000  # 地球半径（单位：km）
    return c * r

def main(args):

    variables = [
        'MAX', 
        'MIN', 
        # # "PRCP",
        "SLP",
        "WDSP",
        # # "STP",
        "MXSPD",
        "DEWP"
    ]

    
    for variable in variables:
        d_max = 20
        folder_path = Path(args.base_path)

        # Load station IDs
        station_data = pd.read_csv(f"{args.base_path}/unique_primary_station_ids_{variable}_tune.csv", dtype={'station_id': str})
        station_ids = station_data['station_id'].tolist()

        # Load CSV files
        csv_files = sorted([
            f for f in (folder_path / variable).glob('*.csv')
        ])

        output_dir = os.path.join(args.base_path, f"pyg_{variable}_{d_max}_graph_step_{args.input_day}_outstep_{args.output_day}")
        os.makedirs(output_dir, exist_ok=True)
        
        for i in tqdm(range(len(csv_files) - args.input_day - args.output_day + 1), desc=f"Processing {variable}"):
            masks = []
            daily_df_input_all = []
            for k in range(args.input_day):
                
                daily_df_input = pd.read_csv(csv_files[i+k], dtype={'station_id': str})
                selected_input_values = torch.from_numpy(daily_df_input.set_index('station_id').reindex(station_ids)['observation_value'].values).to(torch.float32)
                masks.append(~torch.isnan(selected_input_values))
                daily_df_input_all.append(selected_input_values)
                # print(csv_files[i+k])

            daily_df_output_all = []
            for k in range(args.output_day):

                daily_df_output = pd.read_csv(csv_files[i+args.input_day+k], dtype={'station_id': str})
                selected_output_values = torch.from_numpy(daily_df_output.set_index('station_id').reindex(station_ids)['observation_value'].values).to(torch.float32)
                masks.append(~torch.isnan(selected_output_values))
                daily_df_output_all.append(selected_output_values)
                # print(csv_files[i+args.input_day+k])

            mask = torch.stack(masks).all(dim=0)  # shape: (N,)
            x = torch.stack(daily_df_input_all,dim=1)[mask]
            y = torch.stack(daily_df_output_all,dim=1)[mask]

            stations = station_data[['station_id', 'latitude', 'longitude']]

            latitudes_all = torch.from_numpy(stations['latitude'].values).to(torch.float32)
            longitudes_all = torch.from_numpy(stations['longitude'].values).to(torch.float32)

            latitudes = latitudes_all[mask]
            longitudes = longitudes_all[mask]


            edge_indices = []
            edge_values = []

            n = len(latitudes)

            for k in range(n):
                lat1, lon1 = latitudes[k], longitudes[k]
                distances = haversine(lon1, lat1, longitudes, latitudes)

                valid_neighbors = distances < d_max
                valid_indices = np.where(valid_neighbors)[0] 

                for neighbor_idx, weight in zip(valid_indices, distances[valid_indices]):
                    edge_indices.append((k, neighbor_idx))
                    edge_values.append(weight)

            edge_indices = torch.tensor(edge_indices).t().contiguous() 
            edge_values = torch.tensor(edge_values, dtype=torch.float32)

            graph = Data(
                x=x,  # Node features (observation values)
                y=y,
                latitudes=latitudes,
                longitudes=longitudes,
                edge_index=edge_indices,  # Edge indices (connections between nodes)
                edge_attr=edge_values,
                mask=mask  # Edge attributes (distances or weights)
            )
            date_str = csv_files[i].name.split('.')[0]
            torch.save(graph, os.path.join(output_dir, f"{date_str}.pt"))

        print(f"Complete data saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_path', type=str, default='/data/zzn/insitu/data',
                        help='path to csv data')
    parser.add_argument('--ratio', type=float, default=0.1,
                        help='path to csv data')
    parser.add_argument('--input_day', type=int, default=3,
                        help='path to csv data')
    parser.add_argument('--output_day', type=int, default=4,
                        help='path to csv data')
    args = parser.parse_args()
    main(args)

