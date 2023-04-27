import numpy as np
from numpy.linalg import norm
import scipy.sparse
import scipy.sparse.linalg

def dynamic_feature(Input_seq, dynamic_flag=2):
    # parameter for sequencial data
    T, dim = Input_seq.shape
    # prepare W
    W = construct_dynamic_matrix(T, dim, dynamic_flag)
    # Y = Wy
    odata = W.dot(Input_seq.flatten())
    # return odata
    return odata.reshape(T, dim*(dynamic_flag+1))

def generalized_MLPG_v2(Input_seq,Cov,dynamic_flag=2):
    T, sddim = Input_seq.shape
    static_dim = sddim//(dynamic_flag+1)
    Output_seq = np.zeros((T,static_dim))
    for i in range(static_dim):
        Input_seq_new = np.hstack((Input_seq[:,i:i+1],Input_seq[:,i+static_dim:i+static_dim+1],Input_seq[:,i+static_dim*2:i+static_dim*2+1]))
        Cov_new = np.zeros(((dynamic_flag+1),(dynamic_flag+1)))
        Cov_new[1,1] = Cov[i,i]
        Cov_new[2,2] = Cov[i+static_dim,i+static_dim]
        Cov_new[3,3] = Cov[i+static_dim*2,i+static_dim*2]
        Output_seq[:,i] = generalized_MLPG(Input_seq_new,Cov_new,dynamic_flag=2)
    return Output_seq

def fast_MLPG_fixlength(Input_seq,Multipler,dynamic_flag=2):
    T, sddim = Input_seq.shape
    return Multipler.dot(Input_seq.flatten()).reshape(T, sddim//(dynamic_flag+1))

def construct_MLPG_filter(T, dim, Cov, dynamic_flag=2):
    # prepare W
    W = construct_dynamic_matrix(T, dim, dynamic_flag)
    # prepare U
    U = scipy.sparse.block_diag([Cov for i in range(T)], format='csr')
    U.eliminate_zeros()
    # calculate W'U
    WU = W.T.dot(U)
    # W'UW
    WUW = WU.dot(W)
    # estimate y = (W'DW)^-1 * W'Dm
    WUWWU = scipy.sparse.linalg.spsolve(
        WUW, WU, use_umfpack=False)
    # return WUWWU
    return WUWWU

def generalized_MLPG(Input_seq,Cov,dynamic_flag=2):
    # parameter for sequencial data
    T, sddim = Input_seq.shape
    # prepare W
    W = construct_dynamic_matrix(T, sddim//(dynamic_flag+1), dynamic_flag)
    # prepare U
    if Cov.ndim > 2:
        U = scipy.sparse.block_diag(Cov, format='csr')
    else:
        U = scipy.sparse.block_diag([Cov for i in range(T)], format='csr')
    U.eliminate_zeros()
    # calculate W'U
    WU = W.T.dot(U)
    # W'UW
    WUW = WU.dot(W)
    # W'Um
    WUm = WU.dot(Input_seq.flatten())
    # estimate y = (W'DW)^-1 * W'Dm
    odata = scipy.sparse.linalg.spsolve(
        WUW, WUm, use_umfpack=False).reshape(T, sddim//(dynamic_flag+1))
    # return odata
    return odata

def construct_dynamic_matrix(T, D, dynamic_flag=2):
    """
    Calculate static and delta transformation matrix

    Parameters
    ----------
    T : scala, `T`
        Scala of time length
    D : scala, `D`
        Scala of the number of dimentsion

    Returns
    -------
    W : array, shape (`2(or3) * D * T`, `D * T`)
        Array of static and delta transformation matrix.
    """

    # generate full W
    DT = D * T
    ones = np.ones(DT)
    col = np.arange(DT)

    if dynamic_flag == 1:
        static = [0, 1, 0]
        delta = [-1.0, 1.0, 0]

        row = np.arange(2 * DT).reshape(2 * T, D)
        static_row = row[::2]
        delta_row = row[1::2]

        data = np.array([   ones * static[0], ones * static[1],ones * static[2], 
                            ones * delta[0], ones * delta[1], ones * delta[2]]).flatten()
        row = np.array([[static_row] * 3,  [delta_row] * 3]).flatten()
        col = np.array([[col - D, col, col + D] * 2]).flatten()
    else:
        static = [0, 1, 0]
        delta = [-0.5, 0, 0.5]
        delta2 = [1, -2, 1]

        row = np.arange(3 * DT).reshape(3 * T, D)
        static_row = row[::3]
        delta_row = row[1::3]
        delta2_row = row[2::3]

        data = np.array([   ones * static[0], ones * static[1],ones * static[2], 
                            ones * delta[0], ones * delta[1], ones * delta[2],
                            ones * delta2[0], ones * delta2[1], ones * delta2[2],]).flatten()
        row = np.array([[static_row] * 3,  [delta_row] * 3, [delta2_row] * 3]).flatten()
        col = np.array([[col - D, col, col + D] * 3]).flatten()

    # remove component at first and end frame
    valid_idx = np.logical_not(np.logical_or(col < 0, col >= DT))

    W = scipy.sparse.csr_matrix(
        (data[valid_idx], (row[valid_idx], col[valid_idx])), shape=((dynamic_flag+1) * DT, DT))
    W.eliminate_zeros()

    return W
