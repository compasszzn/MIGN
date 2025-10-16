import os
import dgl
import pandas as pd
import torch
import numpy as np
from datetime import datetime, timedelta
import glob
import healpy as hp
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
import argparse
import pickle
from pathlib import Path

def main(args):
    node_types = [
        'MAX', 
        'MIN', 
        'DEWP', 
        "SLP",
        "WDSP",
        "MXSPD",
    ]

    node_data = {}


    for node_type in node_types:
        node_data[node_type] = {}
        
        node_info_df = pd.read_csv(f'{args.base_path}/unique_primary_station_ids_{node_type}_tune.csv')
        station_ids = node_info_df['station_id'].tolist()
        num_stations = len(station_ids)

        
        
        file_paths = Path(os.path.join(args.base_path, node_type))
        csv_files = sorted([
            f for f in file_paths.glob('*.csv') 
        ])
        feature_matrix = np.full((num_stations, len(csv_files)), np.nan)

        for date_idx,file in enumerate(tqdm(csv_files)):

            daily_df = pd.read_csv(file, dtype={'station_id': str})
            daily_df.set_index('station_id', inplace=True)

            for station_idx, station_id in enumerate(station_ids):
                if station_id in daily_df.index:
                    feature_matrix[station_idx, date_idx] = daily_df.loc[station_id, 'observation_value']
                        

        node_data[node_type]['mean'] = np.nanmean(feature_matrix)
        node_data[node_type]['std'] = np.nanstd(feature_matrix)

    with open(args.output_path, 'wb') as f:
        pickle.dump(node_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--base_path', type=str, default='/mnt/hda/zzn/realtime/nips',
                        help='path to csv data')
    parser.add_argument('--output_path', type=str, default='/mnt/hda/zzn/realtime/nips/climatology.npy',
                        help='path to csv data')
    args = parser.parse_args()
    main(args)
