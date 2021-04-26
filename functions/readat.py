import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat


def read_dat(path,):
    """
    path: file path
    """

    dt = np.uint16
    nelements= -1    ### to access all the elements

    with open(path, "rb") as f:
        data_array = np.fromfile(f, dt, nelements, ) #offset=1 
        data_array.shape = (nelements, 1)
        
        fhr_sig = data_array[::2]
        utp_sig = data_array[1::2]

    mTel = 0  
    xdata = []     
    while mTel < len(fhr_sig):
        xdata.append(mTel/240)
        mTel += 1

    fhr_sig = fhr_sig/100
    utp_sig = utp_sig/100

    return fhr_sig, utp_sig, np.array(xdata)

def read_mat(path):
    signal_real = loadmat(path)
    signal_real = signal_real['signal']
    mat_fhr = signal_real[:,0]
    mat_utp = signal_real[:,1]

    mTel = 0 
    x = []      
    while mTel < len(mat_fhr):
        x.append(mTel/240)
        mTel += 1

    return mat_fhr, mat_utp, np.array(x)