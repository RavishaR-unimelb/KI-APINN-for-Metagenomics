import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score
from scipy.spatial.distance import braycurtis
from scipy.stats import linregress

# define model

class PropertyPrediction(nn.Module):

    def __init__(self, in_dim, h_dim, device):
        super(PropertyPrediction, self).__init__()
        self.device = device
        self.input_size  = in_dim
        self.hidden_size = h_dim
        self.output_size = in_dim

        self.lstm    = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear  = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, future=0, y=None):
        outputs = []
        b_size, seq_len, f_size = inputs.shape

        if y is not None:
            # how many steps to predict autoregressively:
            # future = 0 #random.randint(0, seq_len-1)
            future = random.randint(1, int(seq_len) // 2)
            # number of steps to predict from known conditions
            limit  = seq_len - future
        else:
            limit = seq_len

        # set the state of LSTM
        h_t = torch.zeros(inputs.size(0), self.hidden_size, dtype=torch.float32).to(self.device)
        c_t = torch.zeros(inputs.size(0), self.hidden_size, dtype=torch.float32).to(self.device)

        # predict using known conditions
        for i, input_t in enumerate(inputs[:, :limit].chunk(inputs.size(1), dim=1)):
            h_t, c_t   = self.lstm(torch.squeeze(input_t,1), (h_t, c_t))
            output     = self.linear(h_t)
            outputs   += [output]

        # predict autoregressively with teacher forcing
        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = torch.squeeze(y[:,[limit+i],:], 1)
            h_t, c_t   = self.lstm(output, (h_t, c_t))
            output     = self.linear(h_t)
            outputs   += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

class gLV_NN_model(nn.Module):

    def __init__(self, in_dim, h_dim, out_dim, device):
        super(gLV_NN_model, self).__init__()
        self.device = device
        self.input_size  = in_dim
        self.hidden_size = h_dim
        self.output_size = out_dim

        self.linear_gLV  = nn.Linear(self.input_size, self.input_size, bias=True)
        self.linear_NN_h = nn.Linear(self.input_size, self.hidden_size)
        self.linear_NN_o = nn.Linear(self.hidden_size, self.output_size)

        # init gLV parameters to avoid exploiding species trajectories
        with torch.no_grad():
            for n, p in self.named_parameters():
                if 'gLV.weight' in n:
                    # matrix of interaction coefficients
                    p.copy_(torch.zeros(in_dim, in_dim).to(device))
                if 'gLV.bias' in n:
                    # vector of growth rates
                    p.copy_(torch.zeros(in_dim).to(device))

    def forward(self, inputs, y=None):
        # inputs should have shape [Batch Size, Time points, n_species+n_outputs]

        # gLV to estimate next step species abundance
        out = torch.relu(inputs[:,:,:self.input_size] + inputs[:,:,:self.input_size]*(self.linear_gLV(inputs[:,:,:self.input_size])))

        if self.output_size > 0:
            # NN to estimate metabolites
            met_out = self.linear_NN_o(torch.relu(self.linear_NN_h(out)))

            # concatenate outputs
            out = torch.cat((out, met_out), -1)

        return out

    def predict(self, inputs, future):
        # inputs should have shape [Batch Size, 1, n_species+n_outputs]
        # ^ only include initial condition

        # auto-regressive prediction of outputs
        outputs = []
        for i in range(future):
            outputs.append(self.forward(inputs))
            inputs = outputs[-1]

        return torch.cat(outputs, 1)

# 3. Create a PyTorch dataloader to handle creation of training batches

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, sys_vars, sys_scaler):

        # save the df with community data
        self.df = df
        self.sys_vars = sys_vars

        # set unique experiments
        self.comms = np.unique(df['Experiments'].values)

        # set scaler
        self.sys_scaler = sys_scaler

    def __getitem__(self, index):
        # get community
        community = self.comms[index]

        # pull community trajectory
        comm_inds = np.in1d(self.df['Experiments'].values, community)
        D = self.df.iloc[comm_inds].sort_values(by='Time', ascending=True)[self.sys_vars].values

        # standardize data
        X = self.sys_scaler.transform(D)

        # pull features data
        X = torch.tensor(X, dtype=torch.float32)

        # return inputs and output
        x  = X[:-1]
        y  = X[1:]
        return x, y

    def __len__(self):
        # total size of your dataset.
        return len(self.comms)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, X, exp_names, sys_scaler):
        # X is matrix with shape [N samples, N time points, Input dimension]
        self.X = X
        # set scaler on X
        self.sys_scaler = sys_scaler
        # set experiment id
        self.exp_names = exp_names

    def __getitem__(self, index):
        X_i = torch.tensor(self.sys_scaler.transform(self.X[index])[0], dtype=torch.float32)
        exp_name = self.exp_names[index]
        return X_i.unsqueeze(0), exp_name

    def __len__(self):
        # total size of your dataset.
        return self.X.shape[0]

### Organize data into pandas df

def format_data(df, sys_vars):

    # get experiment names
    experiments = df.Experiments.values

    # get unique experiments and number of time measurements
    unique_exps, counts = np.unique(experiments, return_counts=True)

    # determine time vector corresponding to longest sampled experiment
    exp_longest = unique_exps[np.argmax(counts)]
    exp_longest_inds = np.in1d(experiments, exp_longest)
    t_eval = df.iloc[exp_longest_inds]['Time'].values

    # initialize data matrix with NaNs
    D = np.empty([len(unique_exps), len(t_eval), len(sys_vars)])
    D[:] = np.nan

    # fill in data for each experiment
    for i,exp in enumerate(unique_exps):
        print(f'exp:{exp}')
        exp_inds  = np.in1d(experiments, exp)
        print(f'exp_inds:{exp_inds}')
        comm_data = df.copy()[exp_inds]

        # store data
        exp_time = comm_data['Time'].values
        sampling_inds = np.in1d(t_eval, exp_time)
        print(f'sampling_inds:{sampling_inds}')
        D[i][sampling_inds] = comm_data[sys_vars].values

    return D, unique_exps

def format_test_data(test_df, sys_vars):
    ### formats test data to only include initial condition ###

    # get experiment names
    experiments = test_df.Experiments.values

    # get unique experiments and number of time measurements
    unique_exps, seq_lengths = np.unique(experiments, return_counts=True)

    # determine unique sequence lengths
    unique_seq_lengths = np.unique(seq_lengths)

    # each batch has same sequence length
    X_batch = []
    seq_length_batch = []
    exp_name_batch = []
    for seq_length in unique_seq_lengths:
        # get all experiments with seq_length number of time points
        exps = unique_exps[seq_lengths==seq_length]
        X_set = np.zeros([len(exps), 1, len(sys_vars)])
        for i, exp in enumerate(exps):
            exp_df = test_df.iloc[experiments==exp].copy()
            exp_df.sort_values(by='Time', inplace=True)
            # pull only the initial condition
            X_set[i,0,:] = exp_df[sys_vars].values[0]
        X_batch.append(X_set)
        # record number of steps to predict forward
        seq_length_batch.append(seq_length-1)
        exp_name_batch.append(exps)

    return X_batch, seq_length_batch, exp_name_batch

# Define training function

def NNtrain(model, trainloader, optimizer, criterion, device, num_epochs):

    # training settings
    lr_decay       = 0.25
    decay_interval = 12
    # scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', verbose=True)

    for epoch in range(num_epochs):

        # decrease learning rate
        if (epoch+1) % decay_interval == 0:
            if epoch <= 99:
                optimizer.param_groups[0]['lr'] *= lr_decay

        # set up model for training
        train_loss = 0.
        # set training mode
        model.train()
        for i, (x, y) in enumerate(trainloader):

            # send data to gpu or cpu RAM
            # input has dimensions (batch size, n inputs)
            x, y = x.to(device), y.to(device)

            # Forward pass
            output = model(x, y=y)
            output = torch.reshape(output, y.shape)

            # zero gradients
            optimizer.zero_grad()

            # Compute loss
            loss = criterion(output, y)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Keep track of training loss
            train_loss += loss.item()

        # # adjust learning rate
        # scheduler.step(train_loss)

        # print progress
        # if (epoch+1)%50==0:
        print("Epoch: {}/{}, Train Loss: {:.5f}".format(epoch+1, num_epochs, train_loss))

def gLVtrain(model, trainloader, optimizer, criterion, device, num_epochs):

    # training settings
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', verbose=True)

    for epoch in range(num_epochs):

        # set up model for training
        train_loss = 0.
        # set training mode
        model.train()
        for i, (x, y) in enumerate(trainloader):

            # send data to gpu or cpu RAM
            # input has dimensions (batch size, n inputs)
            x, y = x.to(device), y.to(device)

            # Forward pass
            output = model(x, y=y)
            output = torch.reshape(output, y.shape)

            # zero gradients
            optimizer.zero_grad()

            # Compute loss
            loss = criterion(output, y)

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            # Keep track of training loss
            train_loss += loss.item()

        # adjust learning rate
        scheduler.step(train_loss)

        # print progress
        # if (epoch+1)%50==0:
        print("Epoch: {}/{}, Train Loss: {:.5f}".format(epoch+1, num_epochs, train_loss))

class UniformMinMaxScaler():

    def __init__(self, minval=0., maxval=1.):
        self.minval = minval
        self.maxval = maxval
        self.range  = maxval-minval

    def fit(self, X):
        # X has dimensions: (N_experiments, N_timepoints, N_variables)
        self.X_min = np.zeros(X.shape[1:])
        self.X_max = np.nanmax(X, axis=(0,1))
        self.X_range = self.X_max - self.X_min
        self.X_range[self.X_range==0.] = 1.
        return self

    def transform(self, X):
        # convert to 0-1 scale
        X_std = (X - self.X_min) / self.X_range
        # scale to set min-max scale
        X_scaled = X_std*self.range + self.minval
        return X_scaled

    def inverse_transform(self, X_scaled):
        X_std = (X_scaled - self.minval)/self.range
        X = X_std*self.X_range + self.X_min
        return X

    def inverse_transform_stdv(self, X_scaled):
        X_std = (X_scaled - self.minval)/self.range
        X = X_std*self.X_range
        return X

class PerTimeMinMaxScaler():

    def __init__(self, minval=0., maxval=1.):
        self.minval = minval
        self.maxval = maxval
        self.range  = maxval-minval

    def fit(self, X):
        # X has dimensions: (N_experiments, N_timepoints, N_variables)
        self.X_min = np.zeros(X.shape[1:])
        self.X_max = np.nanmax(X, axis=0)
        self.X_range = self.X_max - self.X_min
        self.X_range[self.X_range==0.] = 1.
        return self

    def transform(self, X):
        # convert to 0-1 scale
        X_std = (X - self.X_min) / self.X_range
        # scale to set min-max scale
        X_scaled = X_std*self.range + self.minval
        return X_scaled

    def inverse_transform(self, X_scaled):
        X_std = (X_scaled - self.minval)/self.range
        X = X_std*self.X_range + self.X_min
        return X

    def inverse_transform_stdv(self, X_scaled):
        X_std = (X_scaled - self.minval)/self.range
        X = X_std*self.X_range
        return X

class TimeSeriesStandardScaler():

    def __init__(self, mean=0., std=1.):
        self.mean=mean
        self.std=std

    def fit(self, X):
        # X has dimensions: (N_experiments, N_timepoints, N_variables)
        self.mean = np.nanmean(X, axis=0)
        self.std  = 4*np.nanstd(X, axis=0)
        # center unchanging inputs to zero.
        self.std[self.std==0.] = 1.
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X*self.std + self.mean

class IdentityScaler():
    def __init__(self):
        pass

    def fit(self, X):
        # X has dimensions: (N_experiments, N_timepoints, N_variables)
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X

### Define RNN CLASS here ###

class LSTM():
    def __init__(self, df, sys_vars, device = 'cuda',
                 hidden_size=4096, batch_size=10, lr=5e-3, iteration=200):
        '''
        df is a dataframe with columns
        ['Experiments', 'Time', 'S_1', ..., 'S_M']

        hidden_size := size of neural network hidden layer
        batch_size  := number of samples to include in each training batch
        '''

        # set device
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        # nf := number of features
        self.nf = len(sys_vars)
        self.sys_vars = np.array(sys_vars)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lr = lr
        self.iteration = iteration
        self.comms = np.unique(df['Experiments'].values)

        # format data and initialize data scalers
        X, exp_names = format_data(df, self.sys_vars)
        # fit scalers to training data
        self.sys_scaler = TimeSeriesStandardScaler().fit(X)
        #self.sys_scaler = PerTimeMinMaxScaler().fit(X)

    def train(self, train_df):

        # initialize the data set loaders
        traindataset = Dataset(train_df, self.sys_vars, self.sys_scaler)
        trainloader = torch.utils.data.DataLoader(dataset=traindataset,
                                                  batch_size=self.batch_size)

        # initialize model
        self.rnn = PropertyPrediction(len(self.sys_vars), self.hidden_size, self.device).to(self.device)
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()

        ### train nn
        NNtrain(self.rnn, trainloader, optimizer, criterion,
                device=self.device, num_epochs=self.iteration)

    def predict(self, test_df):
        # init dataframe to return
        return_df = pd.DataFrame()

        # format test data
        X_batch, seq_length_batch, exp_name_batch = format_test_data(test_df, self.sys_vars)

        # initialize the data set loaders
        with torch.no_grad():
            for X, seq_length, exp_names in zip(X_batch, seq_length_batch, exp_name_batch):
                testdataset = TestDataset(X, exp_names, self.sys_scaler)
                testloader  = torch.utils.data.DataLoader(dataset=testdataset, batch_size=self.batch_size)
                for X_i, exps in testloader:
                    # send to device
                    X_i = X_i.to(self.device)
                    # make future predictions
                    X_pred = self.rnn(X_i, future=seq_length-1)
                    # set initial value
                    X_pred = torch.cat((X_i, X_pred), dim=1)
                    # inverse scale
                    X_pred = self.sys_scaler.inverse_transform(X_pred.cpu())
                    # save to dataframe
                    for i, exp in enumerate(exps):
                        test_df_exp = test_df.iloc[test_df.Experiments.values==exp].copy()
                        test_times = test_df_exp['Time'].values
                        exp_df = pd.DataFrame()
                        exp_df['Experiments'] = [exp]*len(test_times)
                        exp_df['Time'] = test_times
                        for j, feature in enumerate(self.sys_vars):
                            exp_df[feature] = test_df_exp[feature].values
                            exp_df[feature + ' pred'] = np.clip(X_pred[i,:,j], 0., np.inf)
                        return_df = pd.concat((return_df, exp_df))
        return return_df

class gLV_NN():
    def __init__(self, df, species, metabolites, device = 'cuda',
                 hidden_size=4096, batch_size=10, lr=5e-3, iteration=200):
        '''
        df is a dataframe with columns
        ['Experiments', 'Time', 'S_1', ..., 'S_M']

        hidden_size := size of neural network hidden layer
        batch_size  := number of samples to include in each training batch
        '''

        # set device
        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')

        # nf := number of features
        self.species = species
        self.metabolites = metabolites
        self.sys_vars = np.concatenate((np.array(species), np.array(metabolites)))
        self.nf = len(self.sys_vars)
        if len(metabolites) > 0:
            self.hidden_size = hidden_size
        else:
            # no need for NN if not predicting metabolites
            self.hidden_size = 1
        self.batch_size = batch_size
        self.lr = lr
        self.iteration = iteration
        self.comms = np.unique(df['Experiments'].values)

        # format data and initialize data scalers
        X, exp_names = format_data(df, self.sys_vars)
        # fit scalers to training data
        self.sys_scaler = PerTimeMinMaxScaler().fit(X)

    def train(self, train_df):

        # initialize the data set loaders
        traindataset = Dataset(train_df, self.sys_vars, self.sys_scaler)
        trainloader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=self.batch_size)

        # initialize model
        self.rnn = gLV_NN_model(len(self.species), self.hidden_size, len(self.metabolites), self.device).to(self.device)
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = torch.nn.MSELoss()

        ### train nn
        gLVtrain(self.rnn, trainloader, optimizer, criterion,
                device=self.device, num_epochs=self.iteration)

    def predict(self, test_df):
        # init dataframe to return
        return_df = pd.DataFrame()

        # format test data
        X_batch, seq_length_batch, exp_name_batch = format_test_data(test_df, self.sys_vars)

        # initialize the data set loaders
        with torch.no_grad():
            for X, seq_length, exp_names in zip(X_batch, seq_length_batch, exp_name_batch):
                testdataset = TestDataset(X, exp_names, self.sys_scaler)
                testloader  = torch.utils.data.DataLoader(dataset=testdataset, batch_size=self.batch_size)
                for X_i, exps in testloader:
                    # send to device
                    X_i = X_i.to(self.device)
                    # make future predictions
                    X_pred = self.rnn.predict(X_i, future=seq_length)
                    # set initial value
                    X_pred = torch.cat((X_i, X_pred), dim=1)
                    # inverse scale
                    X_pred = self.sys_scaler.inverse_transform(X_pred.cpu())
                    # save to dataframe
                    for i, exp in enumerate(exps):
                        test_df_exp = test_df.iloc[test_df.Experiments.values==exp].copy()
                        test_times = test_df_exp['Time'].values
                        exp_df = pd.DataFrame()
                        exp_df['Experiments'] = [exp]*len(test_times)
                        exp_df['Time'] = test_times
                        for j, feature in enumerate(self.sys_vars):
                            exp_df[feature] = test_df_exp[feature].values
                            exp_df[feature + ' pred'] = np.clip(X_pred[i,:,j], 0., np.inf)
                        return_df = pd.concat((return_df, exp_df))
        return return_df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

regular = False
simulated = False
baranwal = False
grier = True
caporaso = False


# set plot parameters
params = {'legend.fontsize': 18,
          'figure.figsize': (16, 12),
          'lines.linewidth': 4,
          'axes.labelsize': 24,
          'axes.titlesize':24,
          'axes.linewidth':5,
          'xtick.labelsize':20,
          'ytick.labelsize':20}
plt.rcParams.update(params)
#plt.style.use('seaborn-colorblind')
plt.rcParams['pdf.fonttype'] = 42

np.random.seed(12345)
###################################################
# For simulated data
if caporaso:
    name = 'M3_feces_L5'
    N = 25
    df = pd.read_csv(f"/data/projects/punim0512/ravisha_projects/Microbial PI-GNN/Data/RNN/rnn_preprocessed_{name}_{N}.csv")
    df = df.drop(columns=["Unnamed: 0"])
    df.head()

    treatments = np.unique(df.Experiments.values)
    #np.random.shuffle(treatments)

    # specify species and metabolite names 
    species = df.columns.values[2:]
    system_variables = np.array(species)
    
    print(f'system_variables: {system_variables}')

    # pull train and test indices
    train_df  = df.copy() 
    test_df   = df.copy() 


if simulated:
    folder_path_train = "/data/projects/punim0512/ravisha_projects/Microbial PI-GNN/Data/Simulated_data/train/"
    folder_path_test = "/data/projects/punim0512/ravisha_projects/Microbial PI-GNN/Data/Simulated_data/test/"

    df_train_list = [f for f in os.listdir(folder_path_train) if f.endswith(".csv")]
    df_test_list = [f for f in os.listdir(folder_path_test) if f.endswith(".csv")]

    train_set_df = []
    test_set_df = []

    for train_name in df_train_list:
        df = pd.read_csv(f'/data/projects/punim0512/ravisha_projects/Microbial PI-GNN/Data/Simulated_data/train/{train_name}')
        train_set_df.append(df)

    for test_name in df_test_list:
        df = pd.read_csv(f'/data/projects/punim0512/ravisha_projects/Microbial PI-GNN/Data/Simulated_data/test/{test_name}')
        test_set_df.append(df)

    
    # List to store modified DataFrames
    train_set_renamed = []
    test_set_renamed = []

    for i, df in enumerate(train_set_df, start=1):
        df_copy = df.copy()
        # Add a suffix to the Experiments column
        df_copy["Experiments"] = df_copy["Experiments"].astype(str) + f"_{i}"
        train_set_renamed.append(df_copy)

    for i, df in enumerate(test_set_df, start=1000):
        df_copy = df.copy()
        # Add a suffix to the Experiments column
        df_copy["Experiments"] = df_copy["Experiments"].astype(str) + f"_{i}"
        test_set_renamed.append(df_copy)
    
    train_df = pd.concat(train_set_renamed, ignore_index=True).fillna(0)
    test_df = pd.concat(test_set_renamed, ignore_index=True).fillna(0)
    print(f'train_df={train_df.shape}')
    print(train_df)

    # Get the full set of columns (union of train + test)
    all_cols = sorted(set(train_df.columns).union(set(test_df.columns)))

    # Reindex both DataFrames — missing columns filled with 0
    train_df = train_df.reindex(columns=all_cols, fill_value=0)
    test_df = test_df.reindex(columns=all_cols, fill_value=0)

    print(train_df)

    species_train = set(train_df.columns.values[2:])  # skip 'Experiments', 'Time'
    species_test = set(test_df.columns.values[2:])
    species = sorted(list(species_train.union(species_test)))
    metabolites =  []
    system_variables = np.concatenate((np.array(species), np.array(metabolites)))
    print(f'system_variables: {system_variables}')


if baranwal:
    df = pd.read_csv("/data/projects/punim0512/ravisha_projects/Microbial PI-GNN/Data/BaranwalClark2022/2021_02_19_MultifunctionalDynamicData_processed.csv")
    df.head()

    # exclude monoculture experiments 
    treatments = np.unique(df.Experiments.values)
    #np.random.shuffle(treatments)

    # Determine index to split last 20%
    split_idx = int(0.8 * len(treatments))

    # Create test set list with last 20% of experiments
    test_set = list(treatments[split_idx:])
    print(f'test_set: {test_set}')

    # specify species and metabolite names 
    species = df.columns.values[2:-4]
    metabolites =  df.columns.values[-4:]
    system_variables = np.concatenate((np.array(species), np.array(metabolites)))
    
    print(f'system_variables: {system_variables}')

    # pull train and test indices
    test_inds = np.in1d(df.Experiments.values, test_set)
    train_df  = df.iloc[~test_inds].copy() 
    test_df   = df.iloc[test_inds].copy() 

    
if regular:
    df = pd.read_csv("/data/projects/punim0512/ravisha_projects/Microbial PI-GNN/Data/Simulated_data/test/Simulated_gLV_data_sparse_10_200.csv")
    df.head()

    # exclude monoculture experiments 
    treatments = np.unique(df.Experiments.values)
    np.random.shuffle(treatments)

    # randomly divide training and testing data (75% training, 25% testing)
    test_set = treatments[:len(treatments)//4]

    # specify species and metabolite names 
    species = df.columns.values[2:]
    metabolites =  []
    system_variables = np.concatenate((np.array(species), np.array(metabolites)))
    print(f'system_variables: {system_variables}')

    # pull train and test indices
    test_inds = np.in1d(df.Experiments.values, test_set)
    train_df  = df.iloc[~test_inds].copy() 
    test_df   = df.iloc[test_inds].copy() 

if grier:
    name = 'NAS_L5'
    df = pd.read_csv(f"/data/projects/punim0512/ravisha_projects/Microbial PI-GNN/Data/raw_grier2018/{name}_preprocessed_rnn.csv")
    df.head()

    treatments = np.unique(df.Experiments.values)
    #np.random.shuffle(treatments)

    # Determine index to split last 20%
    split_idx = int(0.8 * len(treatments))

    # Create test set list with last 20% of experiments
    test_set = list(treatments[split_idx:])
    print(f'test_set: {test_set}')

    # specify species and metabolite names 
    species = df.columns.values[2:]
    system_variables = np.array(species)
    
    print(f'system_variables: {system_variables}')

    # pull train and test indices
    test_inds = np.in1d(df.Experiments.values, test_set)
    train_df  = df.iloc[~test_inds].copy() 
    test_df   = df.iloc[test_inds].copy() 

###################################################

lstm = LSTM(train_df, sys_vars = system_variables, iteration=200)

# fit to data 
lstm.train(train_df)

# test model
k_fold_df = lstm.predict(test_df)

print(f'test shape={test_df}')
print(f'predicted shape: {k_fold_df}')

# Sort and align both DataFrames
test_sorted = test_df.sort_values(by=["Experiments", "Time"]).reset_index(drop=True)
pred_sorted = k_fold_df.sort_values(by=["Experiments", "Time"]).reset_index(drop=True)
print(f'sorted test shape={test_sorted}')
print(f'sorted predicted shape: {pred_sorted}')

if test_sorted["Experiments"].equals(pred_sorted["Experiments"]):
    print("Both DataFrames have identical experiment order.")
else:
    print("Experiment mismatch detected!")
    test_exps = set(test_sorted["Experiments"])
    pred_exps = set(pred_sorted["Experiments"])
    print("Only in test:", test_exps - pred_exps)
    print("Only in predicted:", pred_exps - test_exps)

#Evaluate
meta_cols = ["Experiments", "Time"]
otus = [col for col in test_sorted.columns if col not in meta_cols]

experiments = test_sorted["Experiments"].unique()
n_timesteps = test_sorted["Time"].nunique()
n_otus = len(otus)
print(f'experiments: {experiments}, n_timesteps:{n_timesteps}, n_otus:{n_otus}')

# Containers for scores
r2_scores = []
bcd_scores = []
r2_per_time_scores = []
pooled_r2_scores = []

def mean_bcd_per_timestep_standard(x_reconstructed, x_real):
        """
        Computes Bray-Curtis Dissimilarity per timepoint (community composition),
        with per-timestep normalization, and returns the mean BCD.
        
        Input shape: (num_otus, num_timesteps)
        """
        # Convert tensors to numpy arrays if necessary
        if hasattr(x_real, "cpu"):
            x_real = x_real.cpu().numpy()
        if hasattr(x_reconstructed, "cpu"):
            x_reconstructed = x_reconstructed.cpu().numpy()

        
        num_timesteps = x_real.shape[1]
        bcd_per_time = []

        for t in range(num_timesteps):
            # Normalize per timestep
            x_real_t = x_real[:, t]
            x_reconstructed_t = x_reconstructed[:, t]

            # Avoid division by zero
            sum_real = x_real_t.sum()
            sum_recon = x_reconstructed_t.sum()
            if sum_real > 0:
                x_real_t = x_real_t / sum_real
            if sum_recon > 0:
                x_reconstructed_t = x_reconstructed_t / sum_recon

            # Compute BCD for this timestep
            bcd = braycurtis(x_real_t, x_reconstructed_t)
            bcd_per_time.append(bcd)

        mean_bcd = np.mean(bcd_per_time)
        return bcd_per_time, mean_bcd

def evaluate_pooled_r2(x_reconstructed, x_real):
    """
    Computes R² per OTU (comparing time series), and the mean R².
    Input shape: (num_otus, num_timesteps)
    """
    if hasattr(x_real, "cpu"):
        x_real = x_real.cpu().numpy()
    if hasattr(x_reconstructed, "cpu"):
        x_reconstructed = x_reconstructed.cpu().numpy()

    

    # Center per community (important)
    x_real = x_real - x_real.mean()
    x_reconstructed = x_reconstructed - x_real.mean()

    r2_global = r2_score(x_real.flatten(), x_reconstructed.flatten())

    return r2_global

# ---- Loop over each community ----
true_all = []
pred_all = []
for exp in experiments:
    df_t = test_sorted[test_sorted["Experiments"] == exp]
    df_p = pred_sorted[pred_sorted["Experiments"] == exp]

    #print(f'df_t:{df_t}, df_p:{df_p}')
    
    # Extract OTU abundance matrices over time
    X_true = df_t[otus].to_numpy()
    X_pred = df_p[[col for col in pred_sorted.columns if col.endswith("pred")]].to_numpy()

    #print(f'X_true:{X_true.shape}, X_pred:{X_pred.shape}')
    #print(f'X_true:{X_true}, X_pred:{X_pred}')

    # Convert both to global space - To make it comparable with our method
    # Global normalization
    x_min = X_true.min()
    x_max = X_true.max()
    # Avoid division by zero
    denom = (x_max - x_min) + 1e-8
    # Normalise both with the *same* min/max
    X_true_norm = (X_true - x_min) / denom
    X_pred_norm = (X_pred - x_min) / denom
    X_true = X_true_norm
    X_pred = X_pred_norm

    # --- R² score for each OTU over time ---
    r2_per_otu = [
        r2_score(X_true[:, i], X_pred[:, i])
        for i in range(X_true.shape[1])
    ]
    r2_mean = np.nanmean(r2_per_otu)
    r2_scores.append(r2_mean)

    # --- R² score for each timestep ---
    r2_per_time = [
        r2_score(X_true[i,:], X_pred[i,:])
        for i in range(X_true.shape[0])
    ]
    r2_mean_time = np.nanmean(r2_per_time)
    r2_per_time_scores.append(r2_mean)

    # --- BCD for the community ---
    X_pred_reshaped = X_pred.T
    X_true_reshaped = X_true.T
    bcd_values_per_time, mean_bcd = mean_bcd_per_timestep_standard(X_pred_reshaped, X_true_reshaped)
    pooled_r2 = evaluate_pooled_r2(X_pred_reshaped, X_true_reshaped)
    bcd_scores.append(mean_bcd)
    pooled_r2_scores.append(pooled_r2)

    # Flatten both tensors - Pearson
    real_flat = X_true_reshaped.flatten()
    pred_flat = X_pred_reshaped.flatten()
    true_all.append(real_flat)
    pred_all.append(pred_flat)

# ---- Final average across all communities ----
avg_r2 = np.nanmean(r2_scores)
avg_bcd = np.nanmean(bcd_scores)
avg_r2_time = np.nanmean(r2_per_time_scores)
avg_pooled_r2 = np.nanmean(pooled_r2_scores)

print("Average R² across communities:", avg_r2, avg_r2_time, r2_scores)
print("Average pooled R² across communities:", avg_pooled_r2)
print("Average BCD accuracy across communities:", 1-avg_bcd)

species_preds = []
species_true  = []

# Pearson R2
# get exp times to exclude t=0 from prediction vs measured - removed
exp_times = k_fold_df.Time.values
inds = exp_times > 0

for i, variable in enumerate(species):
    y = k_fold_df[variable].values
    y_pred = k_fold_df[variable+" pred"].values

    # ignore known initial condition 
    species_preds += list(y_pred)
    species_true  += list(y)
    
# Compute R using all predictions (does not include known initial conditions)
species_preds = np.array(species_preds) 
species_true  = np.array(species_true)
m, b, R, R_pval, std_err = linregress(species_preds, species_true)
R2 = R**2
print(f'R2:{R2}')


# Our pearson R2 method - global
# Concatenate across all communities
true_all = np.concatenate(true_all)
pred_all = np.concatenate(pred_all)
_, _, r, _, _ = linregress(pred_all, true_all)
pearson_r2_global = r**2
print(f"Global Pearson R² across all communities (our method): {pearson_r2_global:.4f}")



