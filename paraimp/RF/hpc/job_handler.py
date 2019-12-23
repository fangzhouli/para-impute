import os

class JobHandler(object):
    """private object, dealing with tasks on slurm"""
    def __init__(self, partition, n_cores, memory, time):
        self.path               = os.path.abspath(os.path.dirname(__file__))
        self.partition          = partition
        self.num_core_each_node = n_cores
        self.memory             = memory
        self.time               = time
        self.tmp_X_file         = self.path + '/runinfo/dat/tmp_X.dat'
        self.shell_script_path  = self.path + '/runinfo/dat/job.sh'

    def get_command_shell(self, x_path, argument_path, result_path):
        """return the command to run job.sh"""
        python_path             = 'python'
        script_path             = self.path + '/job.py'
        return ([python_path, script_path, x_path, argument_path, result_path])

    def get_command(self, node_id, job_id, iter_id):
        """return the command of sbatch"""
        exe_path                = 'sbatch'
        par_quiet               = '--quiet'
        par_num_node            = '-N'
        num_node                = '1'
        par_num_core_each_node  = '-c'
        par_memory              = '--mem'
        par_time_limit          = '--time'
        par_job_name            = '-J'
        job_name                = 'impute'
        par_output              = '-o'
        output_ext              = '.out'
        par_error               = '-e'
        error_ext               = '.err'
        shell_script_path       = self.shell_script_path

        if self.partition == None:
            par_partition       = ''
            partition           = ''
        else:
            par_partition       = '-p'
            partition           = self.partition

        num_core_each_node      = str(self.num_core_each_node)
        memory                  = str(self.memory)
        time_limit              = self.time
        job_name                = job_name  + str(node_id) + '_' + str(job_id)\
                                    + '_' + str(iter_id)
        output_file             = self.path + '/runinfo/out/' + job_name + output_ext
        error_file              = self.path + '/runinfo/err/' + job_name + error_ext
        shell_script_path       = shell_script_path

        command = [exe_path, par_quiet, par_partition, partition, par_num_node,
                   num_node,par_num_core_each_node, num_core_each_node,
                   par_memory, memory, par_time_limit, time_limit,par_job_name,
                   job_name, par_output, output_file, par_error, error_file,
                   shell_script_path]

        return ([cmd for cmd in command if cmd != ''])

    def get_arguments_varidx_file(self, node_id, job_id):
        return self.path + '/runinfo/dat/arguments_' + str(node_id) + '_' + str(job_id) + '.dat'

    def get_results_varidx_file(self, node_id, job_id):
        return self.path + '/runinfo/dat/results_' + str(node_id) + '_' + str(job_id) + '.dat'
