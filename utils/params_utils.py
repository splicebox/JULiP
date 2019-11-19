from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

####################################### parameters with default settings #########################
def get_params():
    params = {}
    params['num_samples'] = 1
    params['lo_beta'] = 1./params['num_samples']/10
    params['lo_alpha'] = 1e-4
    params['hi_alpha'] = 3
    params['lr'] = 1
    params['reg_beta'] = 1
    params['reg_alpha'] = 0.1
    params['decay'] = 0.6
    params['minibatch_size']  =  None
    params['n_epochs'] = 1000
    params['patience'] = 2
    params['minibatch_size'] = 80

    params['data_directory'] = "./"
    params['bam_file_list'] = params['data_directory'] + "bamFileList_2.txt"

    params['pickle_file'] = 'results.pkl'

    params['seed'] = 1234
    params['updater'] = 'adam'
    params['save_dir'] = './'
    params['time_stop'] = 24*60*31

    return params


def set_params():
	pass

