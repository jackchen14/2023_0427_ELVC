# -*- coding: utf-8 -*-

import numpy as np


class Statistics(object):
    """F0 statistics class
    Estimate F0 statistics and convert F0

    """

    def __init__(self):
        pass

    def estimate(self, list):
        """Estimate mcep statistics from list of mcep

        Parameters
        ---------
        f0list : list, shape('mcepnum')
            List of several mcep sequence

        Returns
        ---------
        f0stats : array, shape(`[mean, std]`)
            Values of mean and standard deviation for logarithmic F0

        """
        n_files = len(list)
        for i in range(n_files):
            feature = list[i]
            nonzero_indices = np.nonzero(feature)
            if i == 0:
                features = feature[nonzero_indices]
            else:
                features = np.r_[features, feature[nonzero_indices]]

        stats = np.array([np.mean(features), np.std(features)])
        '''
        n_files = len(mceplist)
        mcepstats = []
        for i in range(n_files):
            mcep = mceplist[i]
            mcepstats.append([np.mean(mcep), np.std(mcep)])
        mcepstats = np.array(mcepstats)
        '''
        return stats

    def convert(self, f, tarstats):
        cv = (f*tarstats[1]) + tarstats[0]

        return cv

