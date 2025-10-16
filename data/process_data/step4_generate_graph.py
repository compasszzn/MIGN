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
        # 'air_temperature_mean', 
        'MIN', 
    ]

    d_max = 20
    for variable in variables:
        complete_data = []
        folder_path = Path(args.base_path)

        # Load station IDs
        station_data = pd.read_csv(f"{args.base_path}/unique_primary_station_ids_{variable}_tune.csv")
        station_ids = station_data['station_id'].tolist()

        # Load CSV files
        csv_files = sorted([
            f for f in (folder_path / variable).glob('*.csv')
            if f.stem[:4] in ["2022","2023","2024"]
        ])

        output_dir = os.path.join(args.base_path, f"Realtime_{variable}_20_graph_2022_2024")
        os.makedirs(output_dir, exist_ok=True)

        for i in tqdm(range(len(csv_files[:-1])), desc=f"Processing {variable}"):
            daily_df_input = pd.read_csv(csv_files[i])
            daily_df_output = pd.read_csv(csv_files[i+1])

            # Check if the file is interpolated or not
            # is_interpolated = file.stem.endswith(f'v1_interpolated_{args.method}')

            selected_input_values = torch.from_numpy(daily_df_input.set_index('station_id').reindex(station_ids)['observation_value'].values).to(torch.float32)
            selected_output_values = torch.from_numpy(daily_df_output.set_index('station_id').reindex(station_ids)['observation_value'].values).to(torch.float32)

            mask = ~torch.isnan(selected_input_values) & ~torch.isnan(selected_output_values) 

            x = selected_input_values[mask]
            y = selected_output_values[mask]

            stations = station_data[['station_id', 'latitude', 'longitude']]

            latitudes_all = torch.from_numpy(stations['latitude'].values).to(torch.float32)
            longitudes_all = torch.from_numpy(stations['longitude'].values).to(torch.float32)

            latitudes = latitudes_all[mask]
            longitudes = longitudes_all[mask]


            edge_indices = []
            edge_values = []

            n = len(latitudes)

            for k in range(n):
                # 当前点坐标
                lat1, lon1 = latitudes[k], longitudes[k]
                # 计算与其他点的距离
                distances = haversine(lon1, lat1, longitudes, latitudes)
                
                # 选择距离小于 d_max 的点作为边
                valid_neighbors = distances < d_max
                valid_indices = np.where(valid_neighbors)[0]  # 获取所有符合条件的点的索引
                

                # 保存为稀疏矩阵的边索引和权重
                for neighbor_idx, weight in zip(valid_indices, distances[valid_indices]):
                    edge_indices.append((k, neighbor_idx))
                    edge_values.append(weight)

            edge_indices = torch.tensor(edge_indices).t().contiguous()  # 转置得到正确格式
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

        print(f"Complete data saved to {output_dir}/{date_str}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_path', type=str, default='/mnt/hda/zzn/realtime/Processed',
                        help='path to csv data')
    parser.add_argument('--ratio', type=float, default=0.1,
                        help='path to csv data')
    args = parser.parse_args()
    main(args)

