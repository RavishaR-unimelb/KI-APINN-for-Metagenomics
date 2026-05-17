import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

class Dataset:

    def __init__(self, df):
        self.df = df

    @classmethod
    def from_csv(cls, csv):
        df = pd.read_csv(csv, index_col=0)
        df = pd.concat([df.iloc[:,0:1], df.iloc[:,7:12]], axis=1)
        return cls(df)

    # def preprocess(self, is_simulated=False):
    #
    #     if is_simulated:
    #         self.df = self.df.drop(['Unnamed: 0'], axis=1)
    #         return
    #
    #     cols = list(self.df)
    #     cols = cols[1:]
    #
    #     # normalizing
    #     for col in cols:
    #         min = self.df[col][self.df[col] > 0].min()
    #         self.df[col] /= min
    #
    #     # removing empty_rows
    #     empty_rows = []
    #     for i in range(0, len(self.df)):
    #         if((self.df.iloc[[i]].sum(axis=1)) == 0.0).bool() :
    #             empty_rows.append(i)
    #
    #     for elem in empty_rows:
    #         self.df = self.df.drop(elem)
    #
    #     # Other changes
    #     self.df = self.df.rename(self.df.iloc[:, 0])
    #     self.df = self.df.drop(['Unnamed: 0'], axis=1)
    #     #if (self.df.index == 'Unclassified').any():
    #     #    self.df = self.df.drop(['Unclassified'], axis=0)
    #
    #     # Get abundance values as a percentage
    #     for col in cols:
    #         sum = self.df[col].sum()
    #         self.df[col] = self.df[col] * 100 / sum

    def visualize(self):
        plt.figure(figsize=(100, 20))

        for r, s in self.df.iterrows():
            plt.plot(s, label=r)
        plt.legend()
        plt.show()

    def visualize_stack(self):

        ys = self.matrix().tolist()
        xs = list(self.df)
        ls = list(self.df.index)

        plt.figure(figsize=(100, 20))

        plt.stackplot(xs, *ys, labels=ls)

        plt.legend()
        plt.show()

    def get_subset_by_mean(self, low=10, high=100):
        df_temp = self.df
        df_temp['mean'] = df_temp.mean(axis = 1)

        df_temp = df_temp[(df_temp['mean'] > low) & (df_temp['mean'] <= high)]

        df_temp = df_temp.drop(['mean'], axis = 1)

        return Dataset(df_temp)

    def get_subset_by_rank(self, num, largest=True, random=False):
        df_temp = self.df
        if random:
            df_temp = df_temp.sample(n=num)
            return Dataset(df_temp)

        df_temp['mean'] = df_temp.mean(axis = 1)
        if largest:
            df_temp = df_temp.nlargest(num, 'mean')
        else:
            df_temp = df_temp.nsmallest(num, 'mean')

        df_temp = df_temp.drop(['mean'], axis = 1)

        return Dataset(df_temp)

    def matrix(self):
        return self.df.to_numpy()

    def get_time(self):
        return self.df.shape[1]

    def get_N(self):
        return self.df.shape[0]