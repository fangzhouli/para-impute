"""TODO
1. Check mem vs memcpu
2. In the very future, make categorical input without onehot-encoder
"""

"""Base imputer object, not intended to direct usage"""

import numpy as np

from ._const import InitImpOpt
from ._const import ParallelOpt

class BaseImputer(object):

    def __init__(
            self,
            # max_iter,
            # init_imp,
            parallel,
            partition,
            n_node,
            n_cpu_per_node,
            mem_size,
            time_limit):
        # self.max_iter       = max_iter
        # self.init_imp       = init_imp
        self.parallel       = parallel
        self.partition      = partition
        self.n_node         = n_node
        self.n_cpu_per_node = n_cpu_per_node
        self.mem_size       = mem_size
        self.time_limit     = time_limit
        self._check_inputs()

    def _check_inputs(self):
        """check all the inputs, terminate if there is any invalid"""

        # if type(self.max_iter) is not int or self.max_iter < 1:
        #     raise ValueError("max_iter must be a positive int")

        # if self.init_imp not in [e.value for e in InitImpOpt]:
        #     raise ValueError("init_imp is not valid")

        if self.parallel not in [e.value for e in ParallelOpt]:
            raise ValueError("parallel is not valid")

        if self.parallel == ParallelOpt.SLURM.value:
            if self.partition is not None and type(self.partition) is not str:
                raise ValueError("partition must be None or str")

            if type(self.n_node) is not int or self.n_node < 1:
                raise ValueError("n_node must be a positive int")

            if type(self.n_cpu_per_node) is not int or self.n_cpu_per_node < 1:
                raise ValueError("n_cpu_per_node must be a positive int")

            # TODO
            # if type(self.mem_size) is not int or self.mem_size < 0:
                # raise ValueError("mem_size must be positive")

            # TODO
            # if type

    def _check_matrix(self, X, cat_var):
        """check and record matrix X's information

        input
            X       : (2d-array)
            cat_var : (list of int) indices of cat col
        attributes
            n_      : num samples
            p_      : num variables
            var_t   : a list of [0, 1], 0 for numerical and 1 for
                        categorical, default all 0
            mis_i   : indices of the missing for each variable
            obs_i   : indices of the observed for each variable"""

        try:
            n, p = np.shape(X)
        except:
            raise ValueError("X is not a matrix")

        if type(cat_var) is not list:
            raise ValueError("cat_var must be a list")
        else:
            for ind in cat_var:
                if type(ind) is not int or ind < 0 or ind >= p:
                    raise ValueError("cat_var contains invalid value")

                col = np.array(X[:, ind])
                for val in col:
                    if not np.isnan(val) and val not in [0, 1]:
                        raise ValueError("cat variable contains non-binary val")

        self.Xmis   = np.array(X)
        self.Ximp   = np.copy(self.Xmis)
        self.n_     = n
        self.p_     = p
        self.var_t  = [0] * p
        self.mis_i  = []
        self.obs_i  = []

        # assign var_t
        for i in cat_var:
            self.var_t[i] = 1

        # assign mis_i and obs_i
        for j in range(self.p_):
            col = self.Xmis[:, j]

            mis_i_col = np.where(np.isnan(col))[0]
            obs_i_col = np.delete(np.arange(self.n_), mis_i_col)
            self.mis_i.append(mis_i_col)
            self.obs_i.append(obs_i_col)


    # def _initial_impute_mean(self):
    #     """impute missing values by mean of columns"""

    #     for j in range(self.p_):
    #         col = self.Ximp[:, j]
    #         mean_col = np.mean(col[self.obs_i[j]])

    #         if self.var_t[j]: # categorical variable
    #             if mean_col < 0.5:
    #                 mean_col = 0.0
    #             else:
    #                 mean_col = 1.0

    #         col[self.mis_i[j]] = mean_col

    # def _initial_impute_zero(self):
    #     """impute missing values by zero"""

    #     for j in range(self.p_):
    #         col = self.Ximp[:, j]
    #         col[self.mis_i[j]] = 0.0

    # def _initial_impute(self, X, cat_var = []):
    #     """initial imputation"""

    #     self._check_matrix(X, cat_var)

    #     if self.init_imp == InitImpOpt.MEAN.value:
    #         self._initial_impute_mean()
    #     elif self.init_imp == InitImpOpt.ZERO.value:
    #         self._initial_impute_zero()
































