import os
import shutil
import warnings

import numpy as np

from ._imp import RFImputerProcessorLocal
from ._imp import RFImputerProcessorSlurm
from .consts import InitialGuessOptions
from .consts import ParallelOptions
from .consts import HiddenDirectories

class RFImputer(object):
    '''RandomForestImputer Class

    Parameters
    __________
    NOTE: Parameters are consisted by RFImputer parameters, RandomForest
    parameters, and SLURM parameters. For RandomForest is implemented in
    scikit-learn, many parameters description will be directly referred to
    [2], [3], [4] (who also uses scikit-learn)

    max_iter : int, optional (default=10)
        The maximum number of iterations in case the convergence is not
        achieved.

    init_imp : string (default='mean')
        The mode of initial imputation during the preprocessing:
        - If 'mean', each missing value will be imputed with mean/mode value
        - If 'zero', each missing value will be imputed with zero

    n_estimators : integer, optional (default=100)
        The number of trees in the forest.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider min_samples_split as the minimum number.
        - If float, then min_samples_split is a fraction and ceil(
        min_samples_split * n_samples) are the minimum number of samples for
        each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node. A split
        point at any depth will only be considered if it leaves at least
        min_samples_leaf training samples in each of the left and right
        branches. This may have the effect of smoothing the model, especially
        in regression.
        - If int, then consider min_samples_leaf as the minimum number.
        - If float, then min_samples_leaf is a fraction and ceil(
        min_samples_leaf * n_samples) are the minimum number of samples for
        each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all the
        input samples) required to be at a leaf node. Samples have equal
        weight when sample_weight is not provided.

    max_features : int, float, string or None, optional (default='sqrt')
        The number of features to consider when looking for the best split:
        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and int(max_features *
        n_features) features are considered at each split.
        - If 'auto', then max_features=sqrt(n_features).
        - If 'sqrt', then max_features=sqrt(n_features) (same as “auto”).
        - If 'log2', then max_features=log2(n_features).
        - If None, then max_features=n_features.
        Note: the search for a split does not stop until at least one valid
        partition of the node samples is found, even if
        it requires to effectively inspect more than max_features features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with max_leaf_nodes in best-first fashion. Best nodes are
        defined as relative reduction in impurity. If None then unlimited
        number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following:

        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)
        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child.

        N, N_t, N_t_R and N_t_L all refer to the weighted sum, if
        sample_weight is passed.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees. If False, the
        whole datset is used to build each tree.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to True, reuse the solution of the previous call to fit and
        add more estimators to the ensemble, otherwise, just fit a whole new
        forest. See the Glossary.

    class_weight : dict, list of dicts, “balanced”, “balanced_subsample” or
    None, optional (default=None)
        Weights associated with classes in the form {class_label: weight}. If
        not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be [{0: 1, 1:
        1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1},
        {2:5}, {3:1}, {4:1}].

        The “balanced” mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as n_samples / (n_classes * np.bincount(y))

        The “balanced_subsample” mode is the same as “balanced” except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    partition : string, optional (default=None)
        SLURM parameter, specify your partition on SLURM. Default is specified
        by the administrator of your HPC

    n_cores : int, optional (default=1)
        The number of cores to process. If parallel == 'local', then n_cores
        is exactly the same as n_jobs of Scikit-learn.Setting n_jobs to -1 on
        local machine will use all available cores. If parallel = 'slurm',
        each node uses n_cores number of cores, and it is no longer available
        to be set to -1.

    n_nodes : int, optional (default=1)
        SLURM parameter, specify how many machines (nodes) to use to process

    node_features : int, optional (default=1)
        SLURM parameter, specify how many variables to run in each node
        concurrently. Set the number as high as possible to minimize the
        overhead of parallelization. However, if you set this number too high,
        it will not guarantee you will use all n_nodes number of nodes.
        Recommended number of this parameter is #features / #n_nodes.

    memory : int, optional (default=2000)
        SLURM parameter. specify how much memory in term of MB to allocate for
        each node.

    time : string, optional (default='1:00:00')
        SLURM parameter, specify the time limit of your process to survive.
        The format should be strictly follow:
        - 'minutes'
        - 'minutes:seconds'
        - 'hours:minutes:seconds'
        - 'days-hours'
        - 'days-hours:minutes'
        - 'days-hours:minutes:seconds'

    parallel : string, optional (default='local')
        - If 'local', impute on local machine
        - If 'slurm', impute in parallel on SLURM machines

    Attributes
    __________
    var_ : list
        A list having the same length as the number of variables. Its elements
        are 1, 0, and 1 for numerical, 0 for categorical

    Methods
    _______
    impute(self, Xmis, cat_var=None)：
        return the imputed dataset

        Parameters
        __________
        Xmis : {array-like}, shape (n_samples, n_features)
            Input data, where 'n_samples' is the number of samples and
            'n_features' is the number of features.

        cat_var : list of ints (default=None)
            Specifying the index of columns of categorical variable.

        Return
        ______
        ximp : {array_like}, shape (n_samples, n_features)
            Acquired after imputing all nan of Xmis.'''
    def __init__(self, max_iter=10, init_imp='mean', n_estimators=100,
                 max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features='sqrt',
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, random_state=None, verbose=0,
                 warm_start=False, class_weight=None, partition=None,
                 n_cores=1, n_nodes=1, node_features=1, memory=2000,
                 time='1:00:00', parallel='local'):
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
        self.partition                  = partition
        self.n_cores                    = n_cores
        self.n_nodes                    = n_nodes
        self.node_features              = node_features
        self.memory                     = memory
        self.time                       = time
        self.parallel                   = parallel

    def get_params_imp(self):
        return {
            'max_iter'  : self.max_iter,
            'init_imp'  : self.init_imp,
            'vart_'     : self.vart_
        }

    def get_params_rf(self):
        return {
            'n_estimators'              : self.n_estimators,
            'max_depth'                 : self.max_depth,
            'min_samples_split'         : self.min_samples_split,
            'min_samples_leaf'          : self.min_samples_leaf,
            'min_weight_fraction_leaf'  : self.min_weight_fraction_leaf,
            'max_features'              : self.max_features,
            'max_leaf_nodes'            : self.max_leaf_nodes,
            'min_impurity_decrease'     : self.min_impurity_decrease,
            'bootstrap'                 : self.bootstrap,
            'n_jobs'                    : self.n_cores,
            'random_state'              : self.random_state,
            'verbose'                   : self.verbose,
            'warm_start'                : self.warm_start,
            'class_weight'              : self.class_weight
        }

    def get_params_slurm(self):
        return {
            'partition'     : self.partition,
            'n_nodes'       : self.n_nodes,
            'n_cores'       : self.n_cores,
            'node_features' : self.node_features,
            'memory'        : self.memory,
            'time'          : self.time
        }


    def _check_inputs(self, Xmis, cat_var):
        """private method, validating all inputs"""
        try:
            n, p = np.shape(Xmis)
            self.vart_ = [1 for _ in range(p)]
        except:
            raise ValueError("Xmis: not a matrix")

        if cat_var == None:
            pass
        elif type(cat_var) != list:
            raise ValueError("cat_var: not a list")
        else:
            for i in cat_var:
                self.vart_[i] = 0
        if type(self.max_iter) != int or self.max_iter < 1:
            raise ValueError("max_iter: not a positive integer")
        if self.init_imp not in [e.value for e in InitialGuessOptions]:
            raise ValueError("init_imp: not one of mean, zero, knn")
        if self.parallel not in [e.value for e in ParallelOptions]:
            raise ValueError("parallel: not one of slurm, local")
        if self.parallel == 'slurm':
            if type(self.n_cores) != int or self.n_cores < 1:
                raise ValueError("n_cores: not a positve integer")
            if type(self.n_nodes) != int or self.n_nodes < 1:
                raise ValueError("n_nodes: not a positive integer")
            if self.n_nodes > p:
                raise ValueError("n_nodes: nodes should be less than variables of dataset")
            if type(self.node_features) != int or self.node_features < 1:
                raise ValueError("node_features: not a positive integer")
            if int(p / self.node_features) < self.n_nodes:
                warnings.warn("too large node_features may cause some nodes inactive", SyntaxWarning)
        else:
            if type(self.n_cores) != int or self.n_cores < -1 or self.n_cores == 0:
                raise ValueError("n_cores: neither a positve integer, nor -1")

        return np.array(Xmis)

    def _init_dirs(self):
        """private method, initialize hidden files"""
        runinfo_path = os.path.abspath(os.path.dirname(__file__)) + '/hpc/runinfo'
        files = [runinfo_path + e.value for e in HiddenDirectories]
        for file in files:
            if os.path.exists(file):
                shutil.rmtree(file)
            os.mkdir(file)

    def impute(self, Xmis, cat_var=None):
        """return imputed matrix-like data"""
        Xmis        = self._check_inputs(Xmis, cat_var)
        mf_params   = self.get_params_imp()
        rf_params   = self.get_params_rf()
        sl_params   = self.get_params_slurm()

        if self.parallel == ParallelOptions.LOCAL.value:
            imputer = RFImputerProcessorLocal(mf_params, rf_params)
        else:
            self._init_dirs()
            imputer = RFImputerProcessorSlurm(mf_params, rf_params, **sl_params)
        imputer._impute(Xmis)

        return imputer.result_matrix
