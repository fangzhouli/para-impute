import numpy as np

from pimpute.imputer._base import BaseImputer

X = np.array(
    [[1, 2, 3, 0],
     [4, 5, 6, 1],
     [7, 8, 9, 1],
     [np.nan, np.nan, np.nan, np.nan]])

def test__impute_mean():

    imputer = BaseImputer(
        max_iter        = 10,
        init_imp        = 'mean',
        parallel        = 'local',
        partition       = None,
        n_node          = None,
        n_cpu_per_node  = None,
        mem_size        = None,
        time_limit      = None)

    imputer.impute_mean(X, [3])

    expected = np.array([4.0, 5.0, 6.0, 1.0])

    if np.array_equal(imputer.Ximp[3], expected):
        print("passed test__impute_mean")
    else:
        print("failed test__impute_mean")

def test__impute_zero():

    imputer = BaseImputer(
        max_iter        = 10,
        init_imp        = 'zero',
        parallel        = 'local',
        partition       = None,
        n_node          = None,
        n_cpu_per_node  = None,
        mem_size        = None,
        time_limit      = None)

    imputer.impute_zero(X, [3])

    expected = np.array([0.0, 0.0, 0.0, 0.0])

    if np.array_equal(imputer.Ximp[3], expected):
        print("passed test__impute_zero")
    else:
        print("failed test__impute_zero")

if __name__ == '__main__':
    test__impute_mean()
    test__impute_zero()