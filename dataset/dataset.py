import dgl
import torch
from torch.utils.data import Dataset
from dgl.dataloading import DataLoader,NeighborSampler
import copy
from pathlib import Path
from torch_geometric.data import Data
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import numpy as np
import os
from torch_geometric.utils import dense_to_sparse
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
import pdb


class TemporalHeterogeneousRealTimeDataset(Dataset):
    def __init__(self, data_dir,years,input_length,output_length,climatology=None,data_args=None):

        self.data_dir = Path(f"{data_dir}/Realtime_regular_2022_2024")
        print("loading...",self.data_dir)
        self.data_args = data_args
        self.years = [str(year) for year in years]
        self.all_file=[]
        self.input_length=input_length
        self.output_length=output_length
        for year in self.years:
            self.all_file.extend(list(self.data_dir.glob(f'*{year}*.bin')))
        with open(f"{data_dir}/climatology_our.npy", 'rb') as f:
            self.climatology = pickle.load(f)
        self.all_file.sort()

    def __len__(self):
        return len(self.all_file)

    def normalize(self,graphs):
        for graph in graphs:
            node_type = self.data_args['feature']
            for t in range(self.input_length+self.output_length):
                
                graph.nodes[node_type].data[f't{t}']=(graph.nodes[node_type].data[f't{t}']-torch.tensor(self.climatology[node_type]['mean']).float())/torch.tensor(self.climatology[node_type]['std']).float()
        return graphs

    def get_time_embedding(self,date_str):
        date = datetime.strptime(date_str, "%Y-%m-%d")

        year = date.year
        
        start_of_year = datetime(year, 1, 1)
        end_of_year = datetime(year + 1, 1, 1)
        total_days_in_year = (end_of_year - start_of_year).days
        
        day_of_year = (date - start_of_year).days
        
        day_ratio = day_of_year / total_days_in_year
        
        time_embedding = [date.year-2000, np.cos(2 * np.pi * day_ratio), np.sin(2 * np.pi * day_ratio)]
        
        return time_embedding

    def get_embeddings_for_next_n_days(self,start_date_str, n_days):
        embeddings = []

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        
        for i in range(n_days):
            current_date = start_date + timedelta(days=i)
            current_date_str = current_date.strftime("%Y-%m-%d")
            embedding = self.get_time_embedding(current_date_str)
            embeddings.append(embedding)
        
        return embeddings

    def __getitem__(self, idx):
        graph,sh_embedding=dgl.load_graphs(str(self.all_file[idx]))
        date_str = re.search(r'(\d{4}-\d{2}-\d{2})', str(self.all_file[idx])).group(0)
        embeddings = torch.tensor(self.get_embeddings_for_next_n_days(date_str,self.input_length+self.output_length), dtype=torch.float32)

        graph=self.normalize(graph)
        return graph[0],embeddings,sh_embedding

class TemporalHeterogeneousDataset(Dataset):
    def __init__(self, data_dir,years,input_length,output_length,climatology=None,data_args=None):

        if output_length==1:
            self.data_dir = Path(f"{data_dir}/dgl_neighbor_{data_args['neighbor']}_step_{input_length}_refine_{data_args['refinement_level']}")
        else:
            self.data_dir = Path(f"{data_dir}/dgl_neighbor_{data_args['neighbor']}_step_{input_length}_outstep_{output_length}_refine_{data_args['refinement_level']}")

        print("laoding...",self.data_dir)
        self.data_args = data_args
        self.years = [str(year) for year in years]
        self.all_file=[]
        self.input_length=input_length
        self.output_length=output_length
        for year in self.years:
            # print(self.years)
            self.all_file.extend(list(self.data_dir.glob(f'*{year}*.bin')))
        with open(f"{data_dir}/climatology.npy", 'rb') as f:
            self.climatology = pickle.load(f)
        self.all_file.sort()

    def __len__(self):
        return len(self.all_file)

    def normalize(self,graphs):
        for graph in graphs:
            node_type = self.data_args['feature']
            for t in range(self.input_length+self.output_length):
                graph.nodes[node_type].data[f't{t}']=(graph.nodes[node_type].data[f't{t}']-torch.tensor(self.climatology[node_type]['mean']).float())/torch.tensor(self.climatology[node_type]['std']).float()
        return graphs

    def get_time_embedding(self,date_str):
        date = datetime.strptime(date_str, "%Y-%m-%d")

        year = date.year
        
        start_of_year = datetime(year, 1, 1)
        end_of_year = datetime(year + 1, 1, 1)
        total_days_in_year = (end_of_year - start_of_year).days
        
        day_of_year = (date - start_of_year).days
        
        day_ratio = day_of_year / total_days_in_year
        
        time_embedding = [date.year-2000, np.cos(2 * np.pi * day_ratio), np.sin(2 * np.pi * day_ratio)]
        
        return time_embedding

    def get_embeddings_for_next_n_days(self,start_date_str, n_days):
        embeddings = []

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

        for i in range(n_days):
            current_date = start_date + timedelta(days=i)
            current_date_str = current_date.strftime("%Y-%m-%d")
            embedding = self.get_time_embedding(current_date_str)
            embeddings.append(embedding)
        
        return embeddings

    def __getitem__(self, idx):
        graph,sh_embedding=dgl.load_graphs(str(self.all_file[idx]))
        date_str = re.search(r'(\d{4}-\d{2}-\d{2})', str(self.all_file[idx])).group(0)
        embeddings = torch.tensor(self.get_embeddings_for_next_n_days(date_str,self.input_length+self.output_length), dtype=torch.float32)
        graph=self.normalize(graph)
        return graph[0],embeddings,sh_embedding

class SpatialTemporalDataset(Dataset):
    def __init__(self,data_dir, input_length,output_length,feature,split,climatology=None):
        self.data_dir = data_dir
        self.input_length = input_length
        self.output_length = output_length
        self.feature = feature
        self.split = split
        edge_indices, values =torch.load(os.path.join(self.data_dir, f"{self.feature}_sparse_adjacency_matrix.pt"))
        # print("num_edge",values.shape)
        with open(f"{data_dir}/climatology.npy", 'rb') as f:
            self.climatology = pickle.load(f)

        X = np.load(os.path.join(self.data_dir, f"{self.feature}_kringing_node_values_complete.npy"))

        X = (X-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']

        X_mask = np.load(os.path.join(self.data_dir, f"{self.feature}_kringing_node_values_mask.npy"))



        if self.split=='train':
            self.X = torch.from_numpy(X)[:,0:2192]
            self.X_mask = torch.from_numpy(X_mask)[:,0:2192]
        elif self.split=='val':
            self.X = torch.from_numpy(X)[:,2192:2557]
            self.X_mask = torch.from_numpy(X_mask)[:,2192:2557]
        elif self.split=='test':
            self.X = torch.from_numpy(X)[:,2557:2922]
            self.X_mask = torch.from_numpy(X_mask)[:,2557:2922]
        elif self.split=='all':
            self.X = torch.from_numpy(X)
            self.X_mask = torch.from_numpy(X_mask)

        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def __len__(self):
        num_time_steps = self.X.shape[1]

        return num_time_steps - (self.input_length + self.output_length) + 1

    def __getitem__(self, idx):
        x = self.X[:, idx:idx + self.input_length].clone().detach()  # Input sequence
        y = self.X[:, idx + self.input_length:idx + self.input_length+self.output_length].clone().detach() # Output sequence
        x_mask = ~self.X_mask[:, idx:idx + self.input_length] 
        y_mask = ~self.X_mask[:, idx + self.input_length:idx + self.input_length+self.output_length] 
        y[y_mask] = float('nan')
        x[x_mask] = 0.0
        # print("####")
        
        # Creating a PyG Data object
        data = Data(
            x=torch.tensor(x, dtype=torch.float32) ,  # Node features (input)
            y=torch.tensor(y, dtype=torch.float32),
            x_mask=self.X_mask[:,idx:idx + self.input_length].int(), # Ground truth (output)
            y_mask=self.X_mask[:, idx + self.input_length:idx + self.input_length+self.output_length].int(), 
            edge_index=torch.tensor(self.edges, dtype=torch.long),  # Edge indices
            edge_attr=torch.tensor(self.edge_weights, dtype=torch.float32)  # Edge weights
        )
        return data

class STRealTimeDataset(Dataset):
    def __init__(self,data_dir, input_length,output_length,feature,split,climatology=None,data_args=None):
        self.input_length = input_length
        self.output_length = output_length
        self.feature = feature
        self.split = split

        self.data_dir = Path(os.path.join(data_dir,f"Realtime_{feature}_20_graph_2022_2024"))
        print("loading",self.data_dir)
        self.all_file=list(self.data_dir.glob(f'*.pt'))
        self.all_file.sort()

        with open(f"{data_dir}/climatology.npy", 'rb') as f:
            self.climatology = pickle.load(f)
        # pdb.set_trace()
        if self.split=='train':
            self.data = self.all_file[:365]
        elif self.split=='val':
            self.data = self.all_file[365:730]
        elif self.split=='test':
            self.data = self.all_file[730:]

    def __len__(self):
        num_time_steps = len(self.data)
        return num_time_steps

    def __getitem__(self, idx):
        data = torch.load(self.data[idx])
        data.x = (data.x-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']
        data.y = (data.y-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']
        return data

class STDataset(Dataset):
    def __init__(self,data_dir, input_length,output_length,feature,split,climatology=None,data_args=None):
        self.input_length = input_length
        self.output_length = output_length
        self.feature = feature
        self.split = split
        self.data_args = data_args

        if output_length==1:
            self.data_dir = Path(f"{data_dir}/pyg_{feature}_20_graph_step_{input_length}")
        else:
            self.data_dir = Path(f"{data_dir}/pyg_{feature}_20_graph_step_{input_length}_outstep_{output_length}")

        print("loading",self.data_dir)
        self.all_file=list(self.data_dir.glob(f'*.pt'))
        self.all_file.sort()

        with open(f"{data_dir}/climatology.npy", 'rb') as f:
            self.climatology = pickle.load(f)

        if self.split=='train':
            self.data = self.all_file[0:2192]
        elif self.split=='val':
            self.data = self.all_file[2192:2557]
        elif self.split=='test':
            self.data = self.all_file[2557:2922]

    def __len__(self):
        num_time_steps = len(self.data)
        return num_time_steps

    def __getitem__(self, idx):
        data = torch.load(self.data[idx])
        data.x = (data.x-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']
        data.y = (data.y-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']

        data.longitudes= data.longitudes.unsqueeze(1)
        data.latitudes = data.latitudes.unsqueeze(1)
        return data

class TimeSeriesDataset(Dataset):
    def __init__(self,data_dir, input_length,output_length,feature,split):
        self.data_dir = data_dir
        self.input_length = input_length
        self.output_length = output_length
        self.feature = feature
        self.split = split

        X = np.load(os.path.join(self.data_dir, f"{self.feature}_node_values_complete.npy"))
        X_mask = np.load(os.path.join(self.data_dir, f"{self.feature}_node_values_mask.npy"))


        if self.split=='train':
            self.X = torch.from_numpy(X)[:,0:2192]
            self.X_mask = torch.from_numpy(X_mask)[:,0:2192]
        elif self.split=='val':
            self.X = torch.from_numpy(X)[:,2192:2557]
            self.X_mask = torch.from_numpy(X_mask)[:,2192:2557]
        elif self.split=='test':
            self.X = torch.from_numpy(X)[:,2557:2922]
            self.X_mask = torch.from_numpy(X_mask)[:,2557:2922]

    def __len__(self):
        num_time_steps = self.X.shape[1]
        
        # Calculate the maximum number of valid indices
        return num_time_steps - (self.input_length + self.output_length) + 1

    def __getitem__(self, idx):
        x = self.X[:, idx:idx + self.input_length].clone().detach()  # Input sequence
        y = self.X[:, idx + self.input_length:idx + self.input_length+self.output_length].clone().detach() # Output sequence
        y_mask = ~self.X_mask[:, idx + self.input_length:idx + self.input_length+self.output_length] 
        y[y_mask] = float('nan')
        x = torch.tensor(x.T, dtype=torch.float32)
        y = torch.tensor(y.T, dtype=torch.float32)
        return x,y
    

class Interpolatedataset(Dataset):
    def __init__(self,data_dir, input_length,output_length,feature,split,mask_ratio=0.2,climatology=None):
        self.data_dir = data_dir
        self.input_length = input_length
        self.output_length = output_length
        self.feature = feature
        self.split = split
        self.mask_ratio = mask_ratio
        X ,edge_indices, values,_ =torch.load(os.path.join(self.data_dir, f"{self.feature}.pt"))
        print("loading",self.data_dir)
        with open("data_interpolate_new/step1/data/climatology.pkl", 'rb') as f:
            self.climatology = pickle.load(f)
        

        X = (X-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']

        if self.split=='train':
            self.X = X[:,0:2192]
        elif self.split=='val':
            self.X = X[:,2192:2557]
        elif self.split=='test':
            self.X = X[:,2557:2922]

        self.edges = edge_indices
        self.edge_weights = values

    def __len__(self):
        num_time_steps = self.X.shape[1]

        return num_time_steps - (self.input_length + self.output_length) + 1

    def __getitem__(self, idx):
        x = self.X[:, idx:idx + self.input_length].clone().detach()
        x_valid_mask = ~torch.isnan(x)#True is the valid node
        valid_x = x [x_valid_mask]
        # x[~x_valid_mask] = 0# make the unvalid node to be zero
        

        rand_mask = 1-(torch.rand_like(valid_x) < self.mask_ratio).int()#0 is the mask node, 1 is the valid node

        masked_valid_x = valid_x * rand_mask

        input_x = x.clone().detach()
        input_x[x_valid_mask]=masked_valid_x

        output_x = x.clone().detach()

        valid_indices = x_valid_mask.nonzero(as_tuple=True)
        output_x[valid_indices] = torch.where(rand_mask.bool(), float('nan'), output_x[valid_indices])

        output_mask = x_valid_mask.int().clone().detach()

        input_mask = x_valid_mask.int().clone().detach()
        input_mask[x_valid_mask] = rand_mask

        input_x = torch.nan_to_num(input_x, nan=0.0)

        data = Data(
            x=torch.tensor(input_x, dtype=torch.float32) ,  # Node features (input)
            y=torch.tensor(output_x, dtype=torch.float32),
            x_mask=input_mask, # Ground truth (output)
            y_mask=output_mask, 
            edge_index=torch.tensor(self.edges, dtype=torch.long),  # Edge indices
            edge_attr=torch.tensor(self.edge_weights, dtype=torch.float32)  # Edge weights
        )
        return data

class InterpolatedatasetNEW(Dataset):
    def __init__(self,data_dir, input_length,output_length,feature,split,mask_ratio=0.2,climatology=None):
        self.data_dir = data_dir
        self.input_length = input_length
        self.output_length = output_length
        self.feature = feature
        self.split = split
        self.mask_ratio = mask_ratio
        graph=torch.load(os.path.join(self.data_dir, f"{self.feature}_graph.pt"))
        print("loading",self.data_dir)
        with open(f"{self.data_dir}/climatology.pkl", 'rb') as f:
            self.climatology = pickle.load(f)
        

        # X = (X-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']

        if self.split=='train':
            self.X = graph[0:2192]
        elif self.split=='val':
            self.X = graph[2192:2557]
        elif self.split=='test':
            self.X = graph[2557:2922]


    def __len__(self):
        num_time_steps = len(self.X)

        return num_time_steps

    def __getitem__(self, idx):
        x = self.X[idx].clone().detach()

        mask = ~torch.isnan(x.test_data)
        output = x.x.clone().detach()
        output[mask] = float('nan')
        x.x = (x.test_data-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']
        x.y = (output-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']
        x.x_mask = mask.int()

        x.x = torch.nan_to_num(x.x, nan=0.0)
        x.x = torch.tensor(x.x, dtype=torch.float32)
        x.y = torch.tensor(x.y, dtype=torch.float32)

        return x
    
class InterpolatedatasetNEW2(Dataset):
    def __init__(self,data_dir, input_length,output_length,feature,split,mask_ratio=0.2,climatology=None):
        self.data_dir = data_dir
        self.input_length = input_length
        self.output_length = output_length
        self.feature = feature
        self.split = split
        self.mask_ratio = mask_ratio
        self.df_all = pd.read_csv(f"/data/zzn/insitu/insitu_daily_filter_7_14_filtered/unique_primary_station_ids_{feature}.csv")
        self.position = torch.tensor(self.df_all[['longitude', 'latitude']].values)

        self.embedding = torch.load(f"{self.data_dir}/{feature}_embeddings.pt")
        output_x,_,_,input_x = torch.load(f"{self.data_dir}/{feature}.pt")
        # print(output_x.shape)
        # print("loading",self.data_dir)
        with open(f"{self.data_dir}/climatology.pkl", 'rb') as f:
            self.climatology = pickle.load(f)
        

        output_x = (output_x-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']
        input_x = (input_x-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']

        if self.split=='train':
            self.X = output_x[:,0:2192]
        elif self.split=='val':
            self.X = output_x[:,2192:2557]
        elif self.split=='test':
            self.X = output_x[:,2557:2922]
            self.Y = input_x


    def __len__(self):
        num_time_steps = self.X.shape[1]

        return num_time_steps

    def __getitem__(self, idx):
        x = self.X[:,idx].clone().detach()
        x_mask = ~torch.isnan(x)
        # print(x_mask.shape)
        position = self.position.clone().detach()
        valied_x = x[x_mask]
        valid_positiion = position[x_mask]
        valid_embedding = self.embedding[x_mask]
        bool_tensor = torch.zeros(len(valied_x), dtype=torch.bool)

        # 随机生成一个位置索引
        random_index = torch.randint(0, len(valied_x), (1,)).item()

        # 将该位置设为 True
        bool_tensor[random_index] = True
        predict_position = torch.tensor(valid_positiion[bool_tensor], dtype=torch.float32)#(1*2)tensor
        predict_embedding = torch.tensor(valid_embedding[bool_tensor], dtype=torch.float32)
        predict_value =torch.tensor( valied_x[bool_tensor], dtype=torch.float32)# (1) tensor

        known_position = valid_positiion[~bool_tensor]#(k*2)tensor
        known_embedding = valid_embedding[~bool_tensor]
        known_value = valied_x[~bool_tensor]# (k) tensor

        distances = torch.norm(predict_position - known_position, dim=1)

        # 找到距离最小的十个索引
        _, closest_indices = torch.topk(distances, 10, largest=False)

        # 获取对应的 position 和 value
        closest_positions = torch.tensor(known_position[closest_indices], dtype=torch.float32)
        closest_embedding = torch.tensor(known_embedding[closest_indices], dtype=torch.float32)
        closest_values = torch.tensor(known_value[closest_indices], dtype=torch.float32)


        return closest_positions,closest_embedding,closest_values,predict_position,predict_embedding,predict_value

