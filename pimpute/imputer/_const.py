from enum import Enum

class InitImpOpt(Enum):

    MEAN    = 'mean'
    ZERO    = 'zero'

class ParallelOpt(Enum):

    LOCAL   = 'local'
    SLURM   = 'slurm'

class MaxFeaturesOpt(Enum):

    SQRT    = 'sqrt'
    # TODO