
import torch
import torch.nn.functional as F
import numpy as np
from fastdtw import fastdtw


def Align_loss(orgdata, tardata, orglen, tarlen, distance='euclidean'):
    device = orgdata.device
    loss = 0.0

    for feat1, feat2, len1, len2 in zip(orgdata, tardata, orglen, tarlen):

        feat1 = feat1[:len1]
        feat2 = feat2[:len2]

        feat1_numpy = feat1.detach().cpu().numpy().copy(order='C')
        feat2_numpy = feat2.detach().cpu().numpy().copy(order='C')

        dtwpath = estimate_twf( feat2_numpy, feat1_numpy, distance=distance, unique=1)

        path1 = torch.from_numpy(dtwpath[0]).long().to(device)
        feat1 = feat1.index_select(dim=0, index=path1)

        loss += (feat1 - feat2).pow(2).sum() / feat2.size(1)

    return loss


# def Align_from_tensor_batch(orgdata, tardata, orgmask, tarmask, distance='euclidean', unique=0):
#     device = orgdata.device

#     orgdata_align = []
#     tardata_align = []
#     data_mask = []

#     batch_len1 = orgdata.size(1)
#     batch_len2 = tardata.size(1)

#     batch_len_align = 0

#     for feat1, feat2, mask1, mask2 in zip(orgdata, tardata, orgmask, tarmask):
#         len1 = mask1.ne(0).sum()
#         len2 = mask2.ne(0).sum()

#         feat1_numpy = feat1[:len1].numpy().copy(order='C')
#         feat2_numpy = feat2[:len2].numpy().copy(order='C')

#         dtwpath = estimate_twf( feat1_numpy, feat2_numpy, distance=distance, unique=unique)

#         path1 = torch.from_numpy(dtwpath[0]).long().to(device)
#         path2 = torch.from_numpy(dtwpath[1]).long().to(device)
#         mask = torch.ones_like(path1).to(device)
#         data_mask.append(mask)

#         if mask.size(0) > batch_len_align:
#             batch_len_align = mask.size(0)

#         path1 = F.pad(path1, (0, batch_len1 - path1.size(0)), 'constant').data
#         path2 = F.pad(path2, (0, batch_len2 - path2.size(0)), 'constant').data
        
#         orgdata_align.append(feat1.index_select(dim=0, index=path1))
#         tardata_align.append(feat2.index_select(dim=0, index=path2))

#     orgdata_align = torch.cat([x.unsqueeze(0) for x in orgdata_align],dim=0)
#     tardata_align = torch.cat([x.unsqueeze(0) for x in tardata_align],dim=0)

#     return orgdata_align, tardata_align, data_mask


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


def min_unique(sp1,sp2,dtwpath,uni_flag=2):
    # uni_flag == 0: Do nothing.
    # uni_flag == 1: Find unique path of sp1 with pair has minimum distance
    # uni_flag == 2: Find unique path of both with pair has minimum distance
    d1 = dtwpath[0]
    d2 = dtwpath[1]
    if uni_flag >= 1:
        C = np.unique(d1); 
        idx_final = []
        for _,U in enumerate(C):
            idx = np.array([i for i in range(len(d1)) if d1[i] == U])
            dist = np.sum((sp1[d1[idx]]-sp2[d2[idx]])**2,axis=-1)
            idx_final += [idx[np.argmin(dist)],] 
        idx_final = np.array(idx_final)
        d1 = d1[idx_final]
        d2 = d2[idx_final]

    if uni_flag >= 2:
        C = np.unique(d2); 
        idx_final = []
        for _,U in enumerate(C):
            idx = np.array([i for i in range(len(d2)) if d2[i] == U])
            dist = np.sum((sp1[d1[idx]]-sp2[d2[idx]])**2,axis=-1)
            idx_final += [idx[np.argmin(dist)],] 
        idx_final = np.array(idx_final)
        d1 = d1[idx_final]
        d2 = d2[idx_final]

    return np.vstack((d1,d2))