import scipy as scp
import numpy as np

def matlab2dict(filepath):
    values = scp.io.loadmat(filepath)['output'][0,0]
    keys = scp.io.loadmat(filepath)['output'].dtype.names
    data = {}
    for val, key in zip(values, keys):
        data[key] = np.squeeze(val)
    return data