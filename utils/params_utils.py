from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

####################################### parameters with default setting #########################
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

    params['data_directory'] = "/home-2/gyang22@jhu.edu/julip/data/samples/"
    params['bam_file_list'] = params['data_directory'] + "bamFileList_2.txt"
    #params['bam_file_list'] = "/home-2/gyang22@jhu.edu/work/projects/HPVOP/Tophat/bamFileList.txt"

    params['pickle_file'] = '667_Geuvadis_results.pkl'

    params['seed'] = 1234
    params['updater'] = 'adam'
    params['save_dir'] = './'
    params['time_stop'] = 24*60*31

    return params


def set_params():
	pass

