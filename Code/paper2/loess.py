from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

import numpy as np


def loessline(ls, span=0.1):
    # type: (list, int) -> np.ndarray

    base = importr('base')
    utils = importr('utils')
    pandas2ri.activate()

    ls_r = robjects.FloatVector(ls)

    r = robjects.r

    df = r.matrix(r.unlist(ls_r), nrow=len(ls_r), byrow=r.T)

    rcode = 'data.frame(%s)' % (df.r_repr())
    df = robjects.r(rcode)

    df.columns = ["val"]
    df['index'] = df.index

    df_r = pandas2ri.py2ri(df)

    rcode = 'loessMod <- loess(val ~ index, data=%s, span=%s)' % (df_r.r_repr(), span)
    loessMod_r = robjects.r(rcode)

    rcode = 'predict(%s)' % (loessMod_r.r_repr())
    smoothed_r = robjects.r(rcode)

    assert isinstance(smoothed_r, np.ndarray)
    return smoothed_r
