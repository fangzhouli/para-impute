# para-impute

The para-impute package is a parallelized missing value imputation Python package specialized for high-performance computing (HPC) environment. Currently, the package includes iterative random forest imputation algorithm, (also known as MissForest in R) [1].

### Installation

```
pip install para-impute
```

## Random Forest Imputer

In order to fully utilize the advantage provided by HPC, the package uses a novel parallelization approach to the missing value imputation task:
- Splitting dataset features into different nodes
- Splitting decision trees of random forest into different cores within each node

Random forest imputer relies on [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor) [2] and [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier) [3] of Scikit-learn, so it is currently not available to directly take categorical variables. Instead, please use [one-hot encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) [5] to transform your dataset. You should also input a list of column indices of categorical variable while fitting missing value datasets (see Methods in API section).

### Pseudocode

```
PROGRAM RFImputer(Xmis)
    N <- nrows(Xmis)
    P <- ncols(Xmis)
    Ximp <- Arrange the columns of Xmis in ascending order of the amount of missing values
    Ximp <- Impute each missing values by the mean of all observed values in the same column

    For each column C of Ximp
        Obsi[C] <- indices of observed values
        Misi[C] <- indices of missing values

    While not meeting stopping criteria, iterate
        Xold <- Copy Ximp
        For each column D of Ximp
            ObsX <- Ximp[Obsi[D], All columns except D]
            ObsY <- Ximp[Obsi[D], D]
            MisX <- Ximp[Misi[D], All columns except D]
            MisY <- RandomForest(X_train=Obs, Y_train=ObsY, X_test=MisX)
            Ximp[Misi[D], D] <- MisY

    return Ximp
```
Note: Stopping criteria is defined as follow: when the first time the difference between the dataset of current and previous iteration increases, it stops the iteration and returns the dataset of previous iteration. The metrics for calculating difference are different for numerical and categorical variables.
- For numerical variables, the difference is calculated by Root Mean Square Error (RMSE):
```
diff = sum((Ximp - Xold) ** 2) / sum(Ximp ** 2)
```
- For categorical variables, the difference is calculated by error rate:
```
diff = count(Ximp!=Xold) / #NA
```
For mixed-type dataset (containing both numerical and categorical variables), either one of differences will trigger the stopping criteria.

### Usage

#### Input

An array-like data structure, with missing values represented by either float('nan') or np.nan:
```python
# Example 1
>>> nan = float('nan')
>>> Xmis = [[1.0, 2.0, 3.0],
            [1.5, nan, 2.0],
            [2.0, 1.0, nan]]

# Example 2
>>> nan = np.nan
>>> Xmis = np.array([[1.0, 2.0, 3.0],
                     [1.5, nan, 2.0],
                     [2.0, 1.0, nan]])
```
#### Output

A Numpy Array having the same shape and the same value, except the missing values, as the input:
```python
# Example 1
>>> from pimpute import RFImputer

>>> imputer = RFImputer(parallel='local')
>>> Ximp = imputer.impute(Xmis)
>>> Ximp
array([[1.  , 2.  , 3.  ],
       [1.5 , 1.51, 2.  ],
       [2.  , 1.  , 2.27]])

# Example 2
>>> Xmis = array([[1. , 2. , 3. , 1. , 0. ],
                  [1.5, nan, 2. , 0. , 1. ],
                  [2. , 1. , nan, nan, nan]])
>>> Ximp = imputer.impute(Xmis, cat_var=[3, 4])
>>> Ximp
array([[1.  , 2.  , 3.  , 1.  , 0.  ],
       [1.5 , 1.52, 2.  , 0.  , 1.  ],
       [2.  , 1.  , 2.45, 0.  , 1.  ]])
```

#### SLURM

If you run on 'slurm' mode, make sure you have accessed in machines that have installed SLURM.
```python
>>> from pimpute import RFImputer

>>> nan = np.nan
>>> Xmis = np.array([[1.0, 2.0, nan],
                     [1.1, 2.2, 3.3],
                     [1.5, nan, 5.0]])
>>> imputer = RFImputer(max_iter=10, n_estimators=100, n_nodes=3, n_cores=10, parallel='slurm')
>>> Ximp = imputer.impute(Xmis)
iteration 1
Submitted batch job 4836926
Submitted batch job 4836927
Submitted batch job 4836928
iteration 2
Submitted batch job 4836929
Submitted batch job 4836930
Submitted batch job 4836931
iteration 3
Submitted batch job 4836932
Submitted batch job 4836933
Submitted batch job 4836934
>>> Ximp
array([[1.  , 2.  , 3.  ],
       [1.5 , 1.6 , 2.  ],
       [2.  , 1.  , 2.2]])
```

## API
```
RFImputer(self, max_iter=10, init_imp='mean', n_estimators=100,
                    max_depth=None, min_samples_split=2, min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0, max_features='sqrt',
                    max_leaf_nodes=None, min_impurity_decrease=0.0,
                    bootstrap=True, random_state=None, verbose=0,
                    warm_start=False, class_weight=None, partition=None,
                    n_cores=1, n_nodes=1, node_features=1, memory=2000,
                    time='1:00:00', parallel='local'):

Parameters
__________
NOTE: Parameters are consisted by RFImputer parameters, RandomForest
parameters, and SLURM parameters. Since RandomForest is implemented in
scikit-learn, many parameters description will be directly referred to [2],
[3], [4] that also use scikit-learn.

max_iter : int, optional (default=10)
    The maximum number of iterations to achieve convergence. [What happens when it passes this? Warning?]

init_imp : string (default='mean')
    The mode of initial imputation during the preprocessing:
    - If 'mean', each missing value will be imputed with mean/mode value
    - If 'zero', each missing value will be imputed with zero

n_estimators : integer, optional (default=100)
    The number of trees in the forest.

max_depth : integer or None, optional (default=None)
    The maximum depth of the tree. If None, then nodes are expanded until all
    leaves are pure or until
    all leaves contain less than min_samples_split samples.

min_samples_split : int, float, optional (default=2)
    The minimum number of samples required to split an internal node:
    - If int, then consider min_samples_split as the minimum number.
    - If float, then min_samples_split is a fraction and ceil(
    min_samples_split * n_samples) are the minimum number of samples for
    each split.

min_samples_leaf : int, float, optional (default=1)
    The minimum number of samples required to be at a leaf node. A split point
    at any depth will only be considered if it leaves at least
    min_samples_leaf training samples in each of the left and right branches.
    This may have the effect of
    smoothing the model, especially in regression.
    - If int, then consider min_samples_leaf as the minimum number.
    - If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf
        * n_samples) are the minimum number of samples for each node.

min_weight_fraction_leaf : float, optional (default=0.)
    The minimum weighted fraction of the sum total of weights (of all the
    input samples) required to be at a leaf node. Samples have equal weight
    when sample_weight is not provided.

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
    partition of the node samples is found, even if it requires to effectively
    inspect more than max_features features.

max_leaf_nodes : int or None, optional (default=None)
    Grow trees with max_leaf_nodes in best-first fashion. Best nodes are
    defined as relative reduction in impurity. If None then unlimited number
    of leaf nodes.

min_impurity_decrease : float, optional (default=0.)
    A node will be split if this split induces a decrease of the impurity
    greater than or equal to this value.

    The weighted impurity decrease equation is the following:

    N_t / N * (impurity - N_t_R / N_t * right_impurity
                        - N_t_L / N_t * left_impurity)
    where N is the total number of samples, N_t is the number of samples at
    the current node, N_t_L is the number of samples in the left child, and
    N_t_R is the number of samples in the right child.

    N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is
    passed.

bootstrap : boolean, optional (default=True)
    Whether bootstrap samples are used when building trees. If False, the
    whole datset is used to build each tree.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator; If
    RandomState instance, random_state is the random number generator; If
    None, the random number generator is the RandomState instance used by
    np.random.

verbose : int, optional (default=0)
    Controls the verbosity when fitting and predicting.

warm_start : bool, optional (default=False)
    When set to True, reuse the solution of the previous call to fit and add
    more estimators to the ensemble, otherwise, just fit a whole new forest.
    See the Glossary.

class_weight : dict, list of dicts, “balanced”, “balanced_subsample” or None,
optional (default=None)
    Weights associated with classes in the form {class_label: weight}. If not
    given, all classes are supposed to have weight one. For multi-output
    problems, a list of dicts can be provided in the same order as the columns
    of y.

    Note that for multioutput (including multilabel) weights should be defined
    for each class of every column in its own dict. For example, for
    four-class multilabel classification weights should be [{0: 1, 1: 1}, {0:
        1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1},
        {4:1}].

    The “balanced” mode uses the values of y to automatically adjust weights
    inversely proportional to class frequencies in the input data as n_samples
    / (n_classes * np.bincount(y))

    The “balanced_subsample” mode is the same as “balanced” except that
    weights are computed based on the bootstrap sample for every tree grown.

    For multi-output, the weights of each column of y will be multiplied.

    Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

partition : string, optional (default=None)
    SLURM parameter, specify your partition on SLURM. Default is specified by
    the administrator of your HPC

n_cores : int, optional (default=1)
    The number of cores to process. If parallel == 'local', then n_cores is
    exactly the same as n_jobs of Scikit-learn. Setting n_jobs to -1 on local
    machine will use all available cores. If parallel = 'slurm', each node
    uses n_cores number of cores, and it is no longer available to be set to
    -1.

n_nodes : int, optional (default=1)
    SLURM parameter, specify how many machines (nodes) to use to process

node_features : int, optional (default=1)
    SLURM parameter, specify how many variables to run in each node
    concurrently. Set the number as high as possible to minimize the overhead
    of parallelization. However, if you set this number too high, it will not
    guarantee you will use all n_nodes number of nodes. Recommended number of
    this parameter is #features / #n_nodes.

memory : int, optional (default=2000)
    SLURM parameter. specify how much memory in term of MB to allocate for
    each node.

time : string, optional (default='1:00:00')
    SLURM parameter, specify the time limit of your process to survive. The
    format should be strictly follow:
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
    A list having the same length as the number of variables. Its elements are
    1, 0, and 1 for numerical, 0 for categorical

Methods
_______
fit_transform(self, xmis, cat_var=None)：
    return the imputed dataset

    Parameters
    __________
    xmis : {array-like}, shape (n_samples, n_features)
        Input data, where 'n_samples' is the number of samples and
        'n_features' is the number of features.

    cat_var : list of ints (default=None)
        Specifying the index of columns of categorical variable.

    Return
    ______
    ximp : {array_like}, shape (n_samples, n_features)
        Acquired after imputing all nan of xmis.

```

## Credits

- [ChengEn Tan](https://github.com/bigghost2054) helped the implementation of parallelization
- [Ilias Tagkoupolos](https://github.com/itagkopoulos) as the project advisor

## Reference

- [1] Stekhoven, Daniel J., and Peter Bühlmann. "MissForest—non-parametric missing value imputation for mixed-type data." Bioinformatics 28.1 (2011): 112-118.
- [2] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
- [3] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
- [4] https://github.com/epsilon-machine/missingpy
- [5] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
