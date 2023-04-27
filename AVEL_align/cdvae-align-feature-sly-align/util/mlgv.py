import numpy as np

def fast_MLGV(Input_seq,GV,LV=None,alpha=1.0):
    # Get Local Varience
    if LV is None:
        LV = np.var(Input_seq,axis=0)
    # Get Mean
    X_mean = np.mean(Input_seq,axis=0)
    # Replace Local Varience with Global Varience
    Output_seq = np.sqrt(GV/LV)*(Input_seq-X_mean) + X_mean
    return alpha*Output_seq + (1-alpha)*Input_seq