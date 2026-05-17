from paper2.dataset import Dataset
from scipy.signal import find_peaks
from statsmodels.graphics.tsaplots import plot_acf
import torch

name_list = ['M3_feces_L4', 'M3_feces_L3', 'M3_feces_L2']

for name in name_list:

    # Load data and preprocess
    #name = 'M3_feces_L6' # 'M3_feces_L5' or F4_feces_L5
    df2 = Dataset.from_csv(f'Data/raw/{name}.txt')
    print(df2.df)
    print(len(df2.df))

    if name.startswith('M3'):
        time_points = 332
    else:
        time_points = 130
    print(f'For {name}, timesteps {time_points}...')

    # Select higher abundance OTUs
    N = 30
    df2 = df2.select_by_rank(count=N)
    print(f'df2.df: {df2.df}')


    # Select time interval
    print(f"First {time_points} Timepoints")
    df3 = df2.time_interval(start=0, delta_t=time_points)
    print(f'df3.df: {df3.df}, {df3.df.shape}')

    #df3.df.to_csv('testdf3.csv', index=False)

    df3.visualise(f'abundance_{name}_{N}_{time_points}.jpg', df3.df, time_points, normalize=True)
    #df3.heat_map(f'heatmap_abundance_{name}_{N}_{time_points}.jpg', df3.df, time_points)
    #df3.heat_map_log(f'heatmap_abundance_log_{name}_{N}_{time_points}.jpg', df3.df, time_points)
    df3.visualize_by_level(f'abundance_{name}_{N}_{time_points}_order.jpg', df3.df, time_points, normalize=True)
    ########################################################

    # Convert data to tensors
    data = torch.tensor(df3.df.values, dtype=torch.float32, requires_grad=True)
    print(f'data: {data}')
    print(f'data shape={data.shape}')


    '''
    # Normalize - local
    data_min = torch.min(data, dim=0)[0]
    data_max = torch.max(data, dim=0)[0]
    print(data_min, data_max)
    data_norm = (data - data_min)/(data_max - data_min + 1e-8)
    print(f'Input data: {data_norm}, {data_norm.shape}')

    # Min-max normalization - per timesteps, across OTUs
    x_min = data.min(dim=0, keepdim=True).values
    x_max = data.max(dim=0, keepdim=True).values
    data_norm = (data - x_min) / (x_max - x_min + 1e-8)

    '''
    # Global normalization
    data_min = torch.min(data)
    data_max = torch.max(data)
    data_norm = (data - data_min) / (data_max - data_min + 1e-8)

    print(f'Global min: {data_min}, max: {data_max}')

    print(f'Normalized data shape: {data_norm.shape}')

    # Save
    #torch.save(data_norm, f'preprocessed_data_local_{name}_{N}_{time_points}.pt')

    #torch.save(data, f'preprocessed_data_raw_{name}_{N}_{time_points}.pt')

    #Current
    torch.save(data_norm, f'Embeddings/preprocessed_data_global_{name}_{N}_{time_points}.pt')