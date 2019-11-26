from enum import Enum
import os
import time
import pickle
import subprocess

import numpy as np

from .hpc import JobHandler
from ._rf import RandomForest
from .consts import ParallelOptions
from .consts import InitialGuessOptions
from ..utils.math import mode

class RFImputerProcessor(object):
    """Private class for processing random forest imputation"""
    def __init__(self, max_iter, init_imp, vart_):
        self.max_iter               = max_iter
        self.init_imp               = init_imp
        self.vart_                  = vart_
        self.vari                   = None
        self.misi                   = None
        self.obsi                   = None
        self.previous_diff          = None
        self.matrix_for_impute      = None
        self.initial_guess_matrix   = None
        self.previous_iter_matrix   = None
        self.cur_iter_matrix        = None
        self.result_matrix          = None

    def check_converge(self):
        p = len(self.vart_)
        numi        = [i for i in range(p) if self.vart_[i] == 1]
        cati        = [i for i in range(p) if self.vart_[i] == 0]
        cur_diff    = [None, None]
        # difference of numerical
        if len(numi) > 0:
            X_old_num       = self.previous_iter_matrix[:, numi]
            X_new_num       = self.cur_iter_matrix[:, numi]
            square_diff_sum = np.sum((X_old_num - X_new_num) ** 2)
            square_sum      = np.sum((X_new_num) ** 2)
            cur_diff[0]     = square_diff_sum / square_sum
        # difference of categorical
        if len(cati) > 0:
            X_old_cat       = self.previous_iter_matrix[:, cati]
            X_new_cat       = self.cur_iter_matrix[:, cati]
            num_differ      = np.sum(X_old_cat != X_new_cat)
            num_mis         = sum([self.misi[i] for i in cati])
            cur_diff[1]     = num_differ / num_mis
        # skip if first iteration
        if self.previous_diff is None:
            self.previous_diff = cur_diff
            return False
        else:
            # neither of numerical or categorical should degenerate
            for i in range(2):
                if      (self.previous_diff[i] != None) and\
                        (self.previous_diff[i] < cur_diff[i]):
                    return True
            self.previous_diff = cur_diff
            return False

    def raw_fill(self):
        """imputation preparation, fill missing values with specified values"""
        Xmis = self.matrix_for_impute
        Ximp = np.copy(Xmis)
        n, p = np.shape(Xmis)

        misn = [] # number of missing for each variable
        misi = [] # indices of missing samples for each variable
        obsi = [] # indices of observations for each variable
        for v in range(p):
            vt = self.vart_[v]
            col = Ximp[:, v]
            var_misi = np.where(np.isnan(col))[0]
            var_obsi = np.delete(np.arange(n), var_misi)
            misn.append(len(var_misi))
            misi.append(var_misi)
            obsi.append(var_obsi)
            if vt == 1: # numerical
                if self.init_imp == InitialGuessOptions.MEAN.value:
                    var_mean = np.mean(col[var_obsi])
                    Ximp[var_misi, v] = np.array([var_mean for _ in range(misn[-1])])
                if self.init_imp == InitialGuessOptions.ZERO.value:
                    Ximp[var_misi, v] = np.array([0 for _ in range(misn[-1])])
            else: # categorical
                if self.init_imp == InitialGuessOptions.MEAN.value:
                    var_mode = mode(col[var_obsi].tolist())
                    Ximp[var_misi, v] = np.array([var_mode for _ in range(misn[-1])])
        vari = np.argsort(misn).tolist()
        self.initial_guess_matrix = Ximp
        self.vari = vari
        self.misi = misi
        self.obsi = obsi

class RFImputerProcessorLocal(RFImputerProcessor):
    """private class, random forest processor subclass for local machine"""
    def __init__(self, mf_params, rf_params):
        super().__init__(**mf_params)
        self.params = rf_params

    def _impute(self, matrix_for_impute):
        """impute dataset and return self"""
        self.matrix_for_impute = matrix_for_impute
        self.raw_fill()

        self.previous_iter_matrix = np.copy(self.initial_guess_matrix)
        self.cur_iter_matrix = np.copy(self.initial_guess_matrix)
        cur_iter = 1

        while True:
            if cur_iter > self.max_iter:
                self.result_matrix = self.previous_iter_matrix
                return

            for var in self.vari:
                p = len(self.vart_)
                vt = self.vart_[var]
                cur_X = self.cur_iter_matrix
                cur_obsi = self.obsi[var]
                cur_misi = self.misi[var]
                if (len(cur_misi) == 0):
                    continue
                p_train = np.delete(np.arange(p), var)
                X_train = cur_X[cur_obsi, :][:, p_train]
                y_train = cur_X[cur_obsi, :][:, var]
                X_test = cur_X[cur_misi, :][:, p_train]
                rf = RandomForest(self.params)
                imp = rf.fit_predict(X_train, y_train, X_test, vt)
                self.cur_iter_matrix[cur_misi, var] = imp

            if self.check_converge() == True:
                self.result_matrix = self.previous_iter_matrix
                return
            else:
                self.previous_iter_matrix = np.copy(self.cur_iter_matrix)
                cur_iter = cur_iter + 1

class RFImputerProcessorSlurmArgumentObject(object):
    """private class, contains parameters needed by jobs

    parameters
    ----------
    rf_job  : object contains random forest model
    vart    : list of variable types
    obsi    : list of lists of indices of observed values for each variable
    misi    : list of lists of ondices of missing values for each variable
    results : object stores the result of finished job on cluster"""
    def __init__(self, rf_obj, vart, vari, obsi, misi):
        self.rf_obj     = rf_obj
        self.vart       = vart
        self.vari       = vari
        self.obsi       = obsi
        self.misi       = misi
        self.results    = RFImputerProcessorSlurmResultObject()

class RFImputerProcessorSlurmResultObject(object):
    """private class, stores the result of finished job on cluster

    parameters
    ----------
    imp_list : list records the batch of variables sent to the node
    done     : boolean indicates the job has been done
    err      : error information if job failed
    time     : duration of a job"""
    def __init__(self):
        self.imp_list   = []
        self.done       = False
        self.err        = None
        self.time       = None

class RFImputerProcessorSlurm(RFImputerProcessor):
    """private class, random forest subclass for SLURM machines"""
    def __init__(
            self, mf_params, rf_params, partition, n_nodes, n_cores,
            node_features, memory, time):
        super().__init__(**mf_params)
        self.params         = rf_params
        self.n_nodes        = n_nodes
        self.node_features  = node_features
        self.handler        = JobHandler(partition, n_cores, memory, time)

    def _impute(self, matrix_for_impute):
        self.matrix_for_impute = matrix_for_impute
        self.raw_fill()

        vari_node = self.split_var()
        self.previous_iter_matrix = np.copy(self.initial_guess_matrix)
        self.cur_iter_matrix = np.copy(self.initial_guess_matrix)
        cur_iter = 1

        while True:
            if cur_iter > self.max_iter:
                self.result_matrix = self.previous_iter_matrix
                return

            for i in range(len(vari_node)):
                cur_X = self.cur_iter_matrix
                x_path = self.handler.tmp_X_file
                with open(x_path, 'wb') as tmp:
                    pickle.dump(cur_X, tmp)
                for j in range(len(vari_node[i])):
                    #Prepare the jobs
                    cur_vari = vari_node[i][j]
                    cur_vart = []
                    cur_obsi = []
                    cur_misi = []
                    for k in range(len(vari_node[i][j])):
                        cur_vart.append(self.vart_[cur_vari[k]])
                        cur_obsi.append(self.obsi[cur_vari[k]])
                        cur_misi.append(self.misi[cur_vari[k]])

                    argument_path = self.handler.get_arguments_varidx_file(i, j)
                    result_path = self.handler.get_results_varidx_file(i, j)
                    rf = RandomForest(self.params)
                    with open(argument_path, 'wb') as tmp:
                        argument_object = RFImputerProcessorSlurmArgumentObject(rf, cur_vart, cur_vari, cur_obsi, cur_misi)
                        pickle.dump(argument_object, tmp)
                    with open(result_path, 'wb') as tmp:
                        # argument_object.results.done = False
                        pickle.dump(argument_object.results, tmp)

                    # write job.sh and submit
                    command_shell = self.handler.get_command_shell(x_path, argument_path, result_path)
                    command_shell =' '.join(command_shell)
                    with open(self.handler.shell_script_path, 'w') as tmp:
                        tmp.writelines('#!/bin/bash\n')
                        tmp.writelines(command_shell)
                    command = self.handler.get_command(i, j, cur_iter)
                    subprocess.call(command)

                finish = False
                finished_ind = [False]*len(vari_node[i])
                # finished_count = 0
                while finish == False:
                    time.sleep(0.1)
                    finish = True
                    for j in range(len(vari_node[i])):
                        if finished_ind[j] == True:
                            continue

                        cur_vari = vari_node[i][j]
                        cur_obsi = []
                        cur_misi = []
                        for k in range(len(vari_node[i][j])):
                            cur_obsi.append(self.obsi[cur_vari[k]])
                            cur_misi.append(self.misi[cur_vari[k]])

                        result_path = self.handler.get_results_varidx_file(i, j)
                        try:
                            with open(result_path,'rb') as tmp:
                                cur_result = pickle.load(tmp)
                                if cur_result.done == False:
                                    finish = False
                                    break
                                else:
                                    for k in range(len(cur_vari)):
                                        self.cur_iter_matrix[cur_misi[k],cur_vari[k]] = cur_result.imp_list[k]
                                    finished_ind[j] = True

                        except Exception as e:
                            finish = False
                            break

            if self.check_converge() == True:
                self.result_matrix = self.previous_iter_matrix
                return

            #Update the previous_iter_matrix
            self.previous_iter_matrix = np.copy(self.cur_iter_matrix)

            cur_iter = cur_iter + 1

    def split_var(self):
        #[NODES,[JOBS,[FEATURE]],]

        vari_node = []
        cur_node_idx = 0
        cur_job_idx = 0

        cur_jobs = []
        cur_vari = []

        for var in self.vari:
            cur_vari.append(var)
            if len(cur_vari) == self.node_features:
                cur_jobs.append(cur_vari)
                cur_vari = []
                if len(cur_jobs) == self.n_nodes:
                    vari_node.append(cur_jobs)
                    cur_jobs = []

        if len(cur_vari) > 0:
            cur_jobs.append(cur_vari)
        if len(cur_jobs) > 0:
            vari_node.append(cur_jobs)

        return vari_node