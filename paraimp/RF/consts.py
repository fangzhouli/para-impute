from enum import Enum

class InitialGuessOptions(Enum):

    MEAN    = "mean"
    ZERO    = "zero"

class ParallelOptions(Enum):

    SLURM   = "slurm"
    LOCAL   = "local"

class HiddenDirectories(Enum):

	OUT 	= '/out'
	ERR 	= '/err'
	DATA 	= '/dat'
