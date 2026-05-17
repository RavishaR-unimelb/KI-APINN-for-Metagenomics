from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import umap
import numpy as np
import pandas as pd


def run_umap(dataframe, metric="seuclidean", n_neighbors=2):
    # type: (pd.Dataframe, pd.DataFrame) -> np.ndarray

    reducer = umap.UMAP(n_neighbors=n_neighbors, metric=metric)
    x = dataframe.loc[:, :].values
    embedding = reducer.fit_transform(x)
    dimred = pd.DataFrame(data=embedding, columns=['1', '2'])
    return dimred
