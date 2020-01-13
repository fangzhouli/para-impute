# TODO
#   after implementing rf, implement impute_column
from ._base import BaseImputer
from ._const import InitImpOpt
from ._const import ParallelOpt
from ._const import MaxFeaturesOpt

class MissForest(BaseImputer):

    def __init__(
            self,
            max_iter                    = 10,
            init_imp                    = InitImpOpt.MEAN.value,
            parallel                    = ParallelOpt.LOCAL.value,
            partition                   = None,
            n_node                      = None,
            n_cpu_per_node              = None,
            mem_size                    = None,
            time_limit                  = None,

            n_estimators                = 100,
            max_depth                   = None,
            min_samples_split           = 2,
            min_samples_leaf            = 1,
            min_weight_fraction_leaf    = 0.0,
            max_features                = MaxFeaturesOpt.SQRT.value,
            max_leaf_nodes              = None,
            min_impurity_decrease       = 0.0,
            bootstrap                   = True,
            random_state                = None,
            verbose                     = 0,
            warm_start                  = False,
            class_weight                = None):

        super().__init__(
            # max_iter,
            # init_imp,
            parallel,
            partition,
            n_node,
            n_cpu_per_node,
            mem_size,
            time_limit)
        self.max_iter                   = max_iter
        self.init_imp                   = init_imp

        self.n_estimators               = n_estimators
        self.max_depth                  = max_depth
        self.min_samples_split          = min_samples_split
        self.min_samples_leaf           = min_samples_leaf
        self.min_weight_fraction_leaf   = min_weight_fraction_leaf
        self.max_features               = max_features
        self.max_leaf_nodes             = max_leaf_nodes
        self.min_impurity_decrease      = min_impurity_decrease
        self.bootstrap                  = bootstrap
        self.random_state               = random_state
        self.verbose                    = verbose
        self.warm_start                 = warm_start
        self.class_weight               = class_weight

    def _initial_impute_mean(self):
        """impute missing values by mean of columns"""

        for j in range(self.p_):
            col = self.Ximp[:, j]
            mean_col = np.mean(col[self.obs_i[j]])

            if self.var_t[j]: # categorical variable
                if mean_col < 0.5:
                    mean_col = 0.0
                else:
                    mean_col = 1.0

            col[self.mis_i[j]] = mean_col

    def _initial_impute_zero(self):
        """impute missing values by zero"""

        for j in range(self.p_):
            col = self.Ximp[:, j]
            col[self.mis_i[j]] = 0.0

    def _initial_impute(self):
        """initial imputation"""

        if self.init_imp == InitImpOpt.MEAN.value:
            self._initial_impute_mean()
        elif self.init_imp == InitImpOpt.ZERO.value:
            self._initial_impute_zero()

    def _impute_column(self, col_i):
        """impute a specified column of Ximp"""

        fit_i_row = self.obs_i[col_i]
        fit_i_col = np.delete(np.arange(self.p_), col_i)

        data_fit = self.Ximp[:, fit_i_col][fit_i_row, :]

        self.mis_i[col_i] = RandomForest()
        col_imp     = self.Ximp[:, col_i]
        col_imp_obs = col_imp[self.obs_i[col_i]]
        col_imp_mis = col_imp[self.mis_i[col_i]]

        col_train = self.Ximp[:, self.obs_i[col_i]]

    def impute(self, X, cat_var = []):
        """"""

        self._check_matrix(X, cat_var)
        self._initial_impute()

        if self.parallel == ParallelOpt.LOCAL.value:
            pass
        elif self.parallel == ParallelOpt.SLURM.value:
            pass



























