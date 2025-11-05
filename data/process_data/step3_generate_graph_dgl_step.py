import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1[:, None]
    dlat = lat2 - lat1[:, None]

    # 使用广播来计算哈弗辛公式
    a = np.sin(dlat / 2)**2 + np.cos(lat1[:, None]) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    r = 6371000/1000000  # 地球半径（米）
    return c * r

def main(args):
    node_types = [
        'MAX', 
        'MIN', 
        # "STP",
        "SLP",
        "WDSP",
        "MXSPD",
        "DEWP",
    ]

    for year in [2017,2018,2019,2020,2021,2022,2023,2024]:
        start_date = datetime(year, 1, 1)
        if year==2024:
            end_date = datetime(year, 12, 31)
        else:
            end_date = datetime(year+1, 1, args.input_day+args.output_day-1)#
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end_date - start_date).days + 1)]
        all_label_dict={}
        distance_cache = {}
        node_data = {}
        num_nodes_dict={}
        edges = {}
        distances = {}
        for node_type in node_types:
            node_data[node_type] = []
            
            node_info_df = pd.read_csv(f'{args.base_path}/unique_primary_station_ids_{node_type}_tune.csv', dtype={'station_id': str})
            station_ids = node_info_df['station_id'].tolist()
            num_stations = len(station_ids)
            
            
            feature_matrix = np.full((num_stations, len(dates)), np.nan)
            
            for date_idx, date in enumerate(tqdm(dates, desc=f"Building {node_type} nodes feature")):
                file_path = os.path.join(args.base_path, node_type, f'{date}.csv')
                try:

                    daily_df = pd.concat([pd.read_csv(f, dtype={'station_id': str}) for f in glob.glob(file_path)])
                    daily_df.set_index('station_id', inplace=True)

                    for station_idx, station_id in enumerate(station_ids):
                        if station_id in daily_df.index:
                            feature_matrix[station_idx, date_idx] = daily_df.loc[station_id, 'observation_value']
                    assert (~torch.isnan(torch.tensor(feature_matrix[:,date_idx]))).sum()==len(daily_df)
                            
                except Exception as e:
                    print(f"No data found for {node_type} on {date}: {e}")
            
            node_data[node_type] = torch.tensor(feature_matrix, dtype=torch.float32)
            node_data[f'{node_type}_latitudes'] = torch.tensor(node_info_df['latitude'].values, dtype=torch.float32)
            node_data[f'{node_type}_longitudes'] = torch.tensor(node_info_df['longitude'].values, dtype=torch.float32)

        nside = 2 ** args.refinement_level
        n_pixels = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(n_pixels))
        latitudes = 90 - np.degrees(theta)
        longitudes = np.degrees(phi) - 180
        num_nodes_dict['healpix']=n_pixels

        for node_type in node_types:
            healpix_node_feature = torch.zeros((n_pixels, len(dates)), dtype=torch.float32)
            node_data[f'healpix_feature_{node_type}'] = healpix_node_feature
        node_data['healpix_latitudes'] = torch.tensor(latitudes, dtype=torch.float32)
        node_data['healpix_longitudes'] = torch.tensor(longitudes, dtype=torch.float32)

        for node_type in node_types:
            latitudes = node_data[f'{node_type}_latitudes'].numpy()
            longitudes = node_data[f'{node_type}_longitudes'].numpy()
            healpix_latitudes = node_data['healpix_latitudes'].numpy()
            healpix_longitudes = node_data['healpix_longitudes'].numpy()
            
            distance_cache[node_type] = haversine(longitudes, latitudes, healpix_longitudes, healpix_latitudes)
        distance_cache['healpix'] = haversine(healpix_longitudes, healpix_latitudes, healpix_longitudes, healpix_latitudes)
        
        for node_type in node_types:
            for t in range(len(dates)):
                edges[(node_type, f't{t}_to_healpix', 'healpix')] = ([], [])
                distances[(node_type, f't{t}_to_healpix', 'healpix')] = []
            for t in range(len(dates)):
                edges[('healpix', f't{t}_to_{node_type}', node_type)] = ([], [])
                distances[('healpix', f't{t}_to_{node_type}', node_type)] = []

        edges[('healpix', 'healpix_message', 'healpix')] = ([], [])
        distances[('healpix', 'healpix_message', 'healpix')] = []

        number_of_nearest_neighbour_k = args.number_of_nearest_neighbour
        nearest_nodes_indices = np.argsort(distance_cache['healpix'], axis=1)[:, 0:number_of_nearest_neighbour_k+1] 
        src_nodes = np.repeat(np.arange(num_nodes_dict['healpix']), number_of_nearest_neighbour_k+1)
        dst_nodes = nearest_nodes_indices.flatten()

        edges[('healpix', 'healpix_message', 'healpix')][0].extend(src_nodes)
        edges[('healpix', 'healpix_message', 'healpix')][1].extend(dst_nodes)
        distances[('healpix', 'healpix_message', 'healpix')]=distance_cache['healpix'][src_nodes,dst_nodes]
        all_mask = {}
            
        for node_type in node_types:
            all_mask[node_type]=[]
             # Move distance_cache to GPU
            distance_cache[node_type] = torch.tensor(distance_cache[node_type], dtype=torch.float32).to('cuda') 
            for t in tqdm(range(len(dates)- args.input_day-args.output_day+1),desc=f"Building {node_type} edge feature"):
                input_node_features = node_data[node_type][:, t:t+args.input_day].to('cuda')
                output_node_features = node_data[node_type][:, t+args.input_day].to('cuda')
                mask = ~torch.isnan(output_node_features)
                for mask_input_day in range(args.input_day):
                    mask = ~torch.isnan(input_node_features[:,mask_input_day]) & mask

                all_mask[node_type].append(mask.cpu())
                num_nodes_dict[f'{node_type}_{t}'] = torch.sum(mask.cpu())

                valid_nodes = torch.nonzero(mask).squeeze()
                node_to_healpix_distances = distance_cache[node_type][mask]

                nearest_nodes_indices = torch.argsort(node_to_healpix_distances, axis=0)[:number_of_nearest_neighbour_k, :]
                nearest_nodes = nearest_nodes_indices.T

                edge_src_nodes = nearest_nodes.reshape(-1)
                edge_dst_nodes = torch.repeat_interleave(torch.arange(nearest_nodes_indices.shape[1], device='cuda'), number_of_nearest_neighbour_k)

                nearest_healpix_indices = torch.argsort(node_to_healpix_distances, axis=1)[:, :number_of_nearest_neighbour_k]
                edge_src_healpix = nearest_healpix_indices.flatten()
                edge_dst_healpix = torch.repeat_interleave(torch.arange(nearest_healpix_indices.shape[0], device='cuda'), number_of_nearest_neighbour_k)

                edges[(node_type, f't{t}_to_healpix', 'healpix')][0].extend(edge_src_nodes.flatten().cpu().numpy())
                edges[(node_type, f't{t}_to_healpix', 'healpix')][1].extend(edge_dst_nodes.flatten().cpu().numpy())
                distances[(node_type, f't{t}_to_healpix', 'healpix')].extend(node_to_healpix_distances[nearest_nodes.flatten(),edge_dst_nodes.flatten()].cpu().numpy())

                edges[('healpix', f't{t}_to_{node_type}', node_type)][0].extend(edge_src_healpix.cpu().numpy())
                edges[('healpix', f't{t}_to_{node_type}', node_type)][1].extend(edge_dst_healpix.cpu().numpy())
                distances[('healpix', f't{t}_to_{node_type}', node_type)].extend(node_to_healpix_distances[edge_dst_healpix,edge_src_healpix ].cpu().numpy())

                del input_node_features,output_node_features,valid_nodes,node_to_healpix_distances,nearest_nodes_indices,nearest_nodes
                del edge_src_nodes, edge_dst_nodes, nearest_healpix_indices, edge_src_healpix, edge_dst_healpix
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        for key in edges:
            edges[key] = (torch.tensor(edges[key][0]), torch.tensor(edges[key][1]))
            distances[key] = torch.tensor(distances[key])

        for i in tqdm(range(len(dates) - args.input_day-args.output_day+1),desc=f"Building each graph"):

            current_date = start_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            output_file = os.path.join(args.base_path,f"dgl_neighbor_{number_of_nearest_neighbour_k}_step_{args.input_day}_refine_{args.refinement_level}", f"input_day_{args.input_day}_output_day_{args.output_day}_{date_str}.bin")

            edges_graph={}
            distances_graph={}
            input_time_steps = args.input_day
            output_time_steps = args.output_day

            for node_type in node_types:
                for t in range(input_time_steps):
                    edges_graph[(node_type, f't{t}_to_healpix', 'healpix')] = edges[(node_type, f't{i}_to_healpix', 'healpix')]
                    distances_graph[(node_type, f't{t}_to_healpix', 'healpix')] = distances[(node_type, f't{i}_to_healpix', 'healpix')]
                for t in range(input_time_steps,input_time_steps+output_time_steps):
                    edges_graph[('healpix', f't{t}_to_{node_type}', node_type)] = edges[('healpix', f't{i}_to_{node_type}', node_type)]
                    distances_graph[('healpix', f't{t}_to_{node_type}', node_type)] = distances[('healpix', f't{i}_to_{node_type}', node_type)]
            
            edges_graph[('healpix', 'healpix_message', 'healpix')] = edges[('healpix', 'healpix_message', 'healpix')]
            distances_graph[('healpix', 'healpix_message', 'healpix')] = distances[('healpix', 'healpix_message', 'healpix')]

            split_num_nodes_dict={}
            for node_type in node_types:
                split_num_nodes_dict[node_type]=num_nodes_dict[f'{node_type}_{i}']
            split_num_nodes_dict['healpix']=num_nodes_dict['healpix']
            hetero_graph = dgl.heterograph(edges_graph,split_num_nodes_dict)

            for etype, dist in distances_graph.items():
                hetero_graph.edges[etype].data['distance'] = dist


            for node_type in node_types:
                for t in range(input_time_steps+output_time_steps):
                    hetero_graph.nodes[node_type].data[f't{t}'] = node_data[node_type][:,t+i][all_mask[node_type][i]].unsqueeze(1)
                    hetero_graph.nodes['healpix'].data[f'{node_type}_t{t}'] = node_data[f'healpix_feature_{node_type}'][:,t+i].unsqueeze(1)
                hetero_graph.nodes[node_type].data['latitude'] = node_data[f'{node_type}_latitudes'][all_mask[node_type][i]].unsqueeze(1)
                hetero_graph.nodes[node_type].data['longitude'] = node_data[f'{node_type}_longitudes'][all_mask[node_type][i]].unsqueeze(1)

            hetero_graph.nodes['healpix'].data['latitude'] = node_data['healpix_latitudes'].unsqueeze(1)
            hetero_graph.nodes['healpix'].data['longitude'] = node_data['healpix_longitudes'].unsqueeze(1)

            dgl.save_graphs(output_file, [hetero_graph],{node_type:all_mask[node_type][i] for node_type in node_types})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/data/zzn/insitu/data',
                        help='path to csv data')
    parser.add_argument('--number_of_nearest_neighbour', type=int, default=10,
                        help='path to csv data')
    parser.add_argument('--refinement_level', type=int, default=3,
                        help='HEALPix refinement_level')
    parser.add_argument('--input_day', type=int, default=1,
                        help='input day')
    parser.add_argument('--output_day', type=int, default=1,
                        help='output day')
    args = parser.parse_args()
    main(args)
