import numpy as np

from pimpute import RFImputer

def test_local_numerical():

    nan = float('nan')
    Xmis_1 = [[1.0, 2.0, 3.0],
              [1.5, nan, 2.0],
              [2.0, 1.0, nan]]

    nan = np.nan
    Xmis_2 = np.array([
        [1.0, 2.0, 3.0],
        [1.5, nan, 2.0],
        [2.0, 1.0, nan]])
    imputer = RFImputer(parallel = 'local')
    try:
        Ximp_1 = imputer.impute(Xmis_1)
        Ximp_2 = imputer.impute(Xmis_2)
        print("Passed test_local_numerical")
        print(Ximp_1)
    except Exception as e:
        print("Failed test_local_numerical")
        print(e)
        exit()

def test_local_categorical():

    nan = np.nan
    Xmis = np.array([
        [1. , 2. , 3. , 1. , 0. ],
        [1.5, nan, 2. , 0. , 1. ],
        [2. , 1. , nan, nan, nan]])
    imputer = RFImputer(parallel = 'local')

    Ximp = imputer.impute(Xmis, cat_var=[3, 4])
    try:
        Ximp = imputer.impute(Xmis, cat_var=[3, 4])
        print("Passed test_local_categorical")
    except Exception as e:
        print("Failed test_local_categorical")
        print(e)
        exit()

def test_slurm():

    nan = np.nan
    Xmis = [[1.0, 2.0, 3.0],
            [1.5, nan, 2.0],
            [2.0, 1.0, nan]]
    imputer = RFImputer(max_iter=10, n_estimators=100, n_nodes=3, n_cores=2, parallel='slurm')
    try:
        Ximp = imputer.impute(Xmis)
        print(Ximp)
        print("Passed test_slurm")
    except Exception as e:
        print("Failed test_slurm")
        print(e)
        exit()

if __name__ == '__main__':

    test_local_numerical()
    test_local_categorical()
    # test_slurm()