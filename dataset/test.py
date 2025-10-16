class Interpolatedataset(Dataset):
    def __init__(self,data_dir, input_length,output_length,feature,split,mask_ratio=0.2,climatology=None):
        self.data_dir = data_dir
        self.input_length = input_length
        self.output_length = output_length
        self.feature = feature
        self.split = split
        self.mask_ratio = mask_ratio
        X ,edge_indices, values,test_X =torch.load(os.path.join(self.data_dir, f"{self.feature}.pt"))
        with open("data_interpolate/step1/data/climatology.pkl", 'rb') as f:
            self.climatology = pickle.load(f)
        

        X = (X-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']
        test_X = (test_X-self.climatology[self.feature]['mean'])/self.climatology[self.feature]['std']

        if self.split=='train':
            self.X = X[:,0:2192]
        elif self.split=='val':
            self.X = X[:,2192:2557]
        elif self.split=='test':
            self.X = X[:,2557:2922]
            self.test_X = test_X
        elif self.split=='all':
            self.X = X

        self.edges = edge_indices
        self.edge_weights = values

    def __len__(self):
        num_time_steps = self.X.shape[1]

        return num_time_steps - (self.input_length + self.output_length) + 1

    def __getitem__(self, idx):
        x = self.X[:, idx:idx + self.input_length].clone().detach()
        x_valid_mask = ~torch.isnan(x)#True is the valid node
        valid_x = x [x_valid_mask]
        if self.split=='test':
            input_x = self.test_X[:, idx:idx + self.input_length]
            output_x = x
        else:
            num_elements = valid_x.numel()
            num_nan = int(num_elements * self.mask_ratio) 

            random_indices = torch.randperm(num_elements)[:num_nan]
            valid_x[random_indices] = float('nan')

            input_x = x.clone().detach()
            input_x[x_valid_mask]=valid_x

            output_x = x.clone().detach()

        input_x_mask = torch.isnan(input_x)
        output_x_mask = torch.isnan(output_x)
        result = torch.zeros_like(input_x, dtype=torch.int)

        result[input_x_mask & output_x_mask] = 0   # 都为 NaN
        result[input_x_mask & ~output_x_mask] = 1  # input_x 为 NaN, output_x 不为 NaN
        result[~input_x_mask & ~output_x_mask] = -1 

        data = Data(
            x=torch.tensor(input_x, dtype=torch.float32) ,  # Node features (input)
            y=torch.tensor(output_x, dtype=torch.float32),
            mask=result,
            edge_index=torch.tensor(self.edges, dtype=torch.long),  # Edge indices
            edge_attr=torch.tensor(self.edge_weights, dtype=torch.float32)  # Edge weights
        )
        return data
