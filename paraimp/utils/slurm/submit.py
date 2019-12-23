import subprocess

class Scheduler(object):

    def __init__(self, n_nodes = 1, n_cores = 1, n_features = 1,
            partition = None, memory = 0, time = 0):
        self.n_nodes    = n_nodes
        self.n_cores    = n_cores
        self.n_features = n_features
        self.partition  = partition
        self.memory     = memory
        self.time       = time

    def submit(self, args_list):
        # arg[1] : path to the script file
        for args in args_list:

        pass