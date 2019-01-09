from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

def plot_list(_list):
    indices = range(len(_list))
    plt.scatter(indices, _list)
    plt.show()