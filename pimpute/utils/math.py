from collections import defaultdict

import numpy as np

def isnan(x):
    """return boolean for checking if value is nan"""

    if isinstance(x, (float, np.floating)):
        return np.isnan(x)
    else:
        return x == 'nan'

def mode(x):
    """return the mode of a list"""

    count = defaultdict(int)
    for item in x:
        count[item] += 1
    return max(count, key=count.get)