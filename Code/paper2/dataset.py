from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Dataset:

    def __init__(self, df):
        self.df = df

    @classmethod
    def from_csv(cls, csv):
        # type: (object) -> object
        #df = pd.read_csv(csv, index_col=0)
        df = pd.read_csv(csv, comment='#', index_col=0, sep='\t', header=0)
        return cls(df)

    def time_interval(self, start, delta_t):
        df_new = self.df
        df_new = df_new.iloc[:, start:start + delta_t]

        return Dataset(df_new)

    def select_by_rank(self, count, rev=False):
        df_temp = self.df

        df_temp['mean'] = df_temp.mean(axis=1)
        if rev:
            df_temp = df_temp.nsmallest(count, 'mean')
        else:
            df_temp = df_temp.nlargest(count, 'mean')

        df_temp = df_temp.drop(['mean'], axis=1)

        return Dataset(df_temp)

    def visualise2(self, path):
        print("Visualised")
        fig = self.df.transpose().plot.area(legend=None)
        plt.title('Microbial abundance for OTUs')
        plt.xlabel('Time')
        plt.ylabel('Microbial Abundance')
        plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.savefig(path, dpi=300, bbox_inches='tight')


    def filter_ko_f(self, taxonomy):
        levels = taxonomy.split(';')
        value = ';'.join([l for l in levels if l.startswith(('o__','f__'))])
        print(value)
        return value

    def get_order(self, taxon_name: str) -> str:
        """
        Extract order-level name from OTU taxonomy string.
        Assumes taxonomy strings look like:
        d__Bacteria; p__Proteobacteria; c__Gammaproteobacteria; o__Xanthomonadales; ...
        """
        if "o__" in taxon_name:
            for part in taxon_name.split(";"):
                if part.strip().startswith("o__"):
                    return part.strip()[3:]
        return "Unclassified"   # fallback if no order found
    
    def visualize_by_level_old(self, path, df, t):
        plt.cla()
        plt.clf()

        df_copy = df.copy()

        # Map index to order level
        df_copy.index = df_copy.index.map(self.get_order)

        # Group by order level and sum abundance
        df_grouped = df_copy.groupby(df_copy.index).sum()

        # Reassign timepoint columns
        df_grouped.columns = range(1, t+1)
        df_grouped = np.log1p(df_grouped)

        # Heatmap
        sns.heatmap(df_grouped, cmap="viridis", yticklabels=True)
        plt.title('Microbial abundance profile (Order level)')
        plt.xlabel("Timesteps")
        plt.ylabel("Orders")

        plt.yticks(fontsize=6)
        plt.xticks(fontsize=6)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')

    def visualize_by_level(self, path, df, t, normalize=False):
        plt.cla()
        plt.clf()

        df_copy = df.copy()

        # Map index to order level
        df_copy.index = df_copy.index.map(self.get_order)

        # Group by order level and sum abundance
        df_grouped = df_copy.groupby(df_copy.index).sum()

        # Reassign timepoint columns
        df_grouped.columns = range(1, t+1)
        #df_grouped = np.log1p(df_grouped)
        if normalize:
            #df_grouped = (df_grouped - df_grouped.min().min()) / (df_grouped.max().max() - df_grouped.min().min())
            df_grouped = df_grouped.div(df_grouped.sum(axis=0), axis=1)

        print("Visualised")
        ax = df_grouped.transpose().plot.area(legend=None, colormap='tab20b')  # Use a colormap to ensure distinct colors
        plt.title('Microbial abundance profile (Order level)')
        plt.xlabel("Timesteps")
        plt.ylabel("Orders")

        # Get unique legend handles and labels
        handles, labels = ax.get_legend_handles_labels()
        print(f'Labels in order:{labels}')
        labels = [label for i, label in enumerate(labels)]

        unique = dict(zip(labels, handles))  # Remove duplicates

        plt.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.xlim(1, t)
        plt.xticks(range(1, t+1))

        plt.yticks(fontsize=10)        # adjust font size for y-axis labels
        plt.xticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
    
    def visualise(self, path, df, time_points, normalize=False):
        df_copy = df.copy()
        df_copy.index = df_copy.index.map(self.filter_ko_f)
        df_copy.columns = range(1, time_points+1)

        if normalize:
            #df_copy = (df_copy - df_copy.min().min()) / (df_copy.max().max() - df_copy.min().min())
            df_copy = df_copy.div(df_copy.sum(axis=0), axis=1)
        print("Visualised")
        ax = df_copy.transpose().plot.area(legend=None, colormap='tab20b')  # Use a colormap to ensure distinct colors
        #ax = df_copy.plot.area(legend=None, colormap='tab20b') 
        plt.title('Microbial abundance')
        plt.xlabel('Time')
        plt.ylabel('Microbial Abundance')
        plt.xlim(1, time_points)
        plt.xticks(range(1, time_points+1))
        plt.yticks(fontsize=10)        # adjust font size for y-axis labels
        plt.xticks(fontsize=10)

        # Get unique legend handles and labels
        handles, labels = ax.get_legend_handles_labels()
        print(f'Labels in order:{labels}')
        labels = [label for i, label in enumerate(labels)]

        unique = dict(zip(labels, handles))  # Remove duplicates

        plt.legend(unique.values(), unique.keys(), bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.savefig(path, dpi=300, bbox_inches='tight')

    
    def heat_map(self, path, df, t):
        plt.cla()
        plt.clf()
        df_copy = df.copy()
        print(df_copy.index)
        df_copy.index = df_copy.index.map(self.filter_ko_f)
        print(df_copy.index)
        df_copy.columns = range(1, t+1)
        sns.heatmap(df_copy, cmap="viridis", yticklabels=True)
        plt.title('Microbial abundance profile')
        plt.xlabel("Timesteps")
        plt.ylabel("OTUs")

        plt.yticks(fontsize=6)        # adjust font size for y-axis labels
        plt.xticks(fontsize=6)
        plt.tight_layout()             # fit everything nicely
        
        plt.savefig(path, dpi=300, bbox_inches='tight')

    def heat_map_log(self, path, df, t):
        plt.cla()
        plt.clf()
        df_copy = df.copy()

        # Map OTU names through your filter function
        df_copy.index = df_copy.index.map(self.filter_ko_f)
        df_copy.columns = range(1, t+1)

        # Apply log scale to abundance values
        # Add +1 (or a small epsilon) to avoid log(0) issues
        df_log = np.log1p(df_copy)   # log(1 + x)

        # Plot heatmap
        sns.heatmap(df_log, cmap="viridis", yticklabels=True)
        plt.title('Microbial abundance profile (log scale)')
        plt.xlabel("Timesteps")
        plt.ylabel("OTUs")

        plt.yticks(fontsize=6)  # adjust font size for y-axis labels
        plt.xticks(fontsize=6)
        plt.tight_layout()

        plt.savefig(path, dpi=300, bbox_inches='tight')
    
    # TODO: remove this
    def autoreload_check(self):
        print("Hello Autoreload World")

    @property
    def matrix(self):
        return self.df.to_numpy()

    @property
    def get_time(self):
        return self.df.shape[1]

    # TODO: rename all the properties
    @property
    def get_N(self):
        return self.df.shape[0]

    @property
    def get_time_points(self):
        return list(self.df)

    @property
    def get_OTUs(self):
        return list(self.df.index)

    def get_OTU_short_names(self):
        _list = list(self.df.index)
        for i in range(len(_list)):
            str = _list[i]
            str = str[str.rfind('_')+1:]
            _list[i] = str
        return _list

    @property
    def __str__(self):
        return str(self.df)
