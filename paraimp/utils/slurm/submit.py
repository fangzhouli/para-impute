import subprocess

from .const import SchedulerConst

class Scheduler(object):

    def __init__(self, n_nodes = 1, n_cores = 1, partition = None, memory = 0,
            time = 0):
        self.n_nodes    = n_nodes
        self.n_cores    = n_cores
        self.partition  = partition
        self.memory     = memory
        self.time       = time

    def get_call_command(self):
        return [
            SchedulerConst.FLAG_NODE,
            self.n_nodes,
            SchedulerConst.FLAG_CORE,
            self.n_cores,
            SchedulerConst.]

    def submit(self, args_list):
        # arg[1] : path to the script file
        for args in args_list:

        pass