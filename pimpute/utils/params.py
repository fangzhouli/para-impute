"""utility functions manipulating object data attributes"""

def get_params(imputer):
    """return general parameters of an imputer object

    input
        an imputer obj
    output:
        a dict of parameters"""
    
    return {
        'max_iter'      : imputer.max_iter,
        'init_imp'      : imputer.init_imp,
        'partition'     : imputer.partition,
        'n_nodes'       : imputer.n_nodes,
        'n_cores'       : imputer.n_cores,
        'memory'        : imputer.memory,
        'time'          : imputer.time
    }

def get_params_rf(imputer):
    """return random forest parameters of an imputer object

    input
        an imputer obj
    output:
        a dict of parameters"""

    return {
        'n_estimators'              : imputer.n_estimators,
        'max_depth'                 : imputer.max_depth,
        'min_samples_split'         : imputer.min_samples_split,
        'min_samples_leaf'          : imputer.min_samples_leaf,
        'min_weight_fraction_leaf'  : imputer.min_weight_fraction_leaf,
        'max_features'              : imputer.max_features,
        'max_leaf_nodes'            : imputer.max_leaf_nodes,
        'min_impurity_decrease'     : imputer.min_impurity_decrease,
        'bootstrap'                 : imputer.bootstrap,
        'n_jobs'                    : imputer.n_cores,
        'random_state'              : imputer.random_state,
        'verbose'                   : imputer.verbose,
        'warm_start'                : imputer.warm_start,
        'class_weight'              : imputer.class_weight
    }