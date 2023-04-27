# -*- coding: utf-8 -*-

import numpy as np
from dtw import dtw
from fastdtw import fastdtw
import ipdb

def estimate_twf(orgdata, tardata, distance='melcd', radius=1, fast=True, unique=0):
    """time warping function estimator

    Parameters
    ---------
    orgdata : array, shape(`T_org`, `dim`)
        Array of source feature
    tardata : array, shape(`T_tar`, `dim`)
        Array of target feature
    distance : str, optional
        distance function
        `melcd` : mel-cepstrum distortion
    fast : bool, optional
        Use fastdtw instead of dtw
        Default set to `True`

    Returns
    ---------
    twf : array, shape(`2`, `T`)
        Time warping function between original and target
    """

    if distance == 'melcd':
        def distance_func(x, y): return melcd(x, y)
    elif distance == 'euclidean':
        def distance_func(x, y): return np.mean((x - y) ** 2)
    elif distance == 'cossim': 
        def distance_func(x, y): return - np.inner(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError('other distance metrics than melcd does not support.')

    if fast:
        _, path = fastdtw(orgdata, tardata, radius=radius,dist=distance_func)
        twf = np.array(path).T
    else:
        # _, _, _, twf = dtw(orgdata, tardata, distance_func)
        # twf = np.array(twf)
        _, path = fastdtw(orgdata, tardata, radius=10000000, dist=distance_func)
        twf = np.array(path).T

    twf = min_unique( orgdata, tardata, twf, uni_flag=unique)

    return twf

def melcd(array1, array2):
    """Calculate mel-cepstrum distortion

    Calculate mel-cepstrum distortion between the arrays.
    This function assumes the shapes of arrays are same.

    Parameters
    ----------
    array1, array2 : array, shape (`T`, `dim`) or shape (`dim`)
        Arrays of original and target.

    Returns
    -------
    mcd : scala, number > 0
        Scala of mel-cepstrum distortion

    """
    if array1.shape != array2.shape:
        raise ValueError(
            "The shapes of both arrays are different \
            : {} / {}".format(array1.shape, array2.shape))

    if array1.ndim == 2:
        # array based melcd calculation
        diff = array1 - array2
        mcd = 10.0 / np.log(10) \
            * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=1)))
    elif array1.ndim == 1:
        diff = array1 - array2
        mcd = 10.0 / np.log(10) * np.sqrt(2.0 * np.sum(diff ** 2))
    else:
        raise ValueError("Dimension mismatch")

    return mcd


def min_unique_half(sp1,sp2,dtwpath):
    d1 = dtwpath[0]
    d2 = dtwpath[1]
    C = np.unique(d1); 
    idx_final = []
    for _,U in enumerate(C):
        idx = np.array([i for i in range(len(d1)) if d1[i] == U])
        dist = np.sum((sp1[d1[idx]]-sp2[d2[idx]])**2,axis=-1)
        idx_final += [idx[np.argmin(dist)],] 
    idx_final = np.array(idx_final)
    d1 = d1[idx_final]
    d2 = d2[idx_final]

    return np.vstack((d1,d2))

def min_unique(sp1,sp2,dtwpath,uni_flag=2):
    # uni_flag == 0: Do nothing.
    # uni_flag == 1: Find unique path of sp1 with pair has minimum distance
    # uni_flag == 2: Find unique path of both with pair has minimum distance
    d1 = dtwpath[0]
    d2 = dtwpath[1]
    # print("uni_flag = " , uni_flag)
    # print("shape b4")
    # print(d1)
    # print(d2)
    # if uni_flag >= 1:
    #     C = np.unique(d1); 
    #     idx_final = []
    #     for _,U in enumerate(C):
    #         idx = np.array([i for i in range(len(d1)) if d1[i] == U])
    #         dist = np.sum((sp1[d1[idx]]-sp2[d2[idx]])**2,axis=-1)
    #         idx_final += [idx[np.argmin(dist)],] 
    #     idx_final = np.array(idx_final)
    #     d1 = d1[idx_final]
    #     d2 = d2[idx_final]

    if uni_flag >= 2:
        C = np.unique(d2); 
        idx_final = []
        for _,U in enumerate(C):
            # print(U)
            idx = np.array([i for i in range(len(d2)) if d2[i] == U])
            # print(idx)
            dist = np.sum((sp1[d1[idx]]-sp2[d2[idx]])**2,axis=-1)
            idx_final += [idx[np.argmin(dist)],]
            # print(idx_final)
        idx_final = np.array(idx_final)
        d1 = d1[idx_final]
        d2 = d2[idx_final]
    # print("shape after")
    # print(d1)
    # print(d2)
    # ipdb.set_trace()

    return np.vstack((d1,d2))