#!/usr/bin/env python

# Job script executed on cluster. Printed inforemation stored in output
# files at .out/ directory. Error information stored in error files at .err/.

from time import time
import sys
import mmap
import pickle

import numpy as np

if __name__ == "__main__":

    start_job = time()
    start_load = time()
    data_file = sys.argv[1]
    arg_file = sys.argv[2]
    res_file = sys.argv[3]
    with open(data_file, "r+b") as tmp:
        mm = mmap.mmap(tmp.fileno(), 0)
        X = pickle.load(mm)
    with open(arg_file, "rb") as tmp:
        arg_obj = pickle.load(tmp)
    print("Duration of load:", time() - start_load)

    imp_list = []
    X_array = np.array(X)
    for i in range(len(arg_obj.vari)):
        start_rf = time()
        print("Start variable", i, '...')
        vart = arg_obj.vart[i]
        vari = arg_obj.vari[i]
        misi = arg_obj.misi[i]
        obsi = arg_obj.obsi[i]
        _, p = np.shape(X_array)

        p_train = np.delete(np.arange(p), vari)
        X_train = X_array[obsi, :][:, p_train]
        y_train = X_array[obsi, :][:, vari]
        X = X_array[misi, :][:, p_train]
        rf = arg_obj.rf_obj
        y = rf.fit_predict(X_train, y_train, X, vart)
        X_array[misi, vari] = y
        imp_list.append(y)
        print("Duration of imputing var", i, ":", time() - start_rf)

    duration_job = time() - start_job
    arg_obj.results.done = True
    arg_obj.results.imp_list = imp_list
    arg_obj.results.time = duration_job

    with open(res_file, "wb") as tmp:
        pickle.dump(arg_obj.results, tmp)
