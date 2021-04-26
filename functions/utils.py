import numpy as np
import pickle
import matplotlib.pyplot as plt
import glob
import os
import math
import numpy as np
from scipy import stats
import collections
import tqdm
from scipy.interpolate import interp1d

def signal_len(signal):
    ''' 
    signal: 'list' any input signal, fhr or xdata to find the length
    returns: 'int' s_duration (duration of the signal)
    '''
    num_samples = len(signal)
    s_duration = math.floor( (num_samples / 4) / 60 )   ### Each ctg signal was sampled at 4 Hz
    return s_duration

def running_mean(x, N):
    """ x == an array of data. N == number of samples per average """
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def linear_interpolate(x, fhr, utp):
    filtered_FHR = np.squeeze(np.array(fhr))
    filtered_UTP = np.squeeze(np.array(utp))
    filtered_x = np.array(x)
    
    f_fhr = interp1d(filtered_x, filtered_FHR)
    f_utp = interp1d(filtered_x, filtered_UTP)

    mTel =  0
    x = []      
    while mTel <= round(filtered_x[-1]) * 240:
        x.append(mTel/240)
        mTel += 1
    #### introduce upper and lower bound for x data
    filtered_ind = np.where(np.array(x) > filtered_x[0])[0]
    x = [ x[ind] for ind in filtered_ind ]
    filtered_ind = np.where(np.array(x) < filtered_x[-1])[0]
    x = [ x[ind] for ind in filtered_ind ]
    #### perform interpolation
    fhr_inter = [ f_fhr( ind ) for ind in x ]
    utp_inter = [ f_utp( ind ) for ind in x ]

    return fhr_inter, utp_inter, x

def get_keys(path_to_files='/home/sband/Kidaho/experiments/dataset/*.hea', duplicate=True):
    """ 
    path_to_files: 'str', path containing file names that would be used as keys,
    duplicate: 'bool', if folder contains multiple files containing the key word
    returns: keys
    """
    path = glob.glob(path_to_files)#
    keys = [os.path.basename(p).split('.')[0] for p in path]
    keys.sort()
    if duplicate == True:
        keys = keys[::2]
    return keys

def load_dict(file_path="wfdbdata_processed.pkl"):
    """
    file_path: 'str', path to the .pkl file or .npy file.
    returns: 'list or dict' file content.
    """
    file_name = os.path.splitext(os.path.basename(file_path))[1]
    if file_name == '.pkl':
        a_file = open(file_path, "rb")
        processed = pickle.load(a_file)
        a_file.close()
    elif file_name == '.npy':
        pass
    return processed

def get_utpcount(utp_signal, winSize, slideInt):
    """
    utp_signal: [array or list], complete utp signal to find the count.
    winSize: [int], Size of the window in terms of number of samples.
    slideInt: sliding interval of window in terms of number samples.

    ##Returns
    utp_count: count of uterine contractions
    utp_bplot:  to plot the utp baseline
    utp_inter: the intersection points between utp baseline and utp signal
    """
    # if isinstance(utp_signal, list):
    utp_signal=  list( map( lambda x: round(x), utp_signal ) )
    utp_signal = np.array(utp_signal)

    samp_utp = utp_signal[:7200]

    # utp_virbase = samp_utp.mean()
    utp_virbase, _ = stats.mode(samp_utp)

    utp_bplot = np.zeros(samp_utp.shape)   ### to store actual utp baseline
    utp_rplot = np.zeros(samp_utp.shape)   ### to store virtual utp baseline
    for utpWin in range(0, 7200-winSize, slideInt):   ### get actual baseline for the give window size and slide interval
        #     print(utpWin)
        utp_mod = samp_utp[utpWin:utpWin+winSize]
        # utp_base = utp_mod.mean()
        utp_base, _ = stats.mode(utp_mod)
    #     print(utp_base)
        utp_bplot[utpWin:utpWin+winSize] = utp_base 

    utp_bplot = list( map( lambda x: int(round(x)), utp_bplot ) )   ### get a smooth baseline curve for comparison
    utp_bplot = np.array(utp_bplot)
    utp_rplot[:7200] = utp_virbase    #### curve for virtual baseline

    for each_int in range(len(samp_utp)):        ### iterate over all values to round up values to deacc_lim for error window +-4
        try_b = samp_utp[each_int] 
        deacc_lim = utp_bplot[each_int]
        if try_b in range(deacc_lim-4, deacc_lim+4):
            samp_utp[each_int] = deacc_lim    ### include neighbouring points in threshold to get relaible intersection of signals
            #print(try_b)

    utp_inter = []    #### to store interscetions with baseline
    iu = 0
    for firstUtp, firstBase in zip(samp_utp[:len(utp_bplot)], utp_bplot):
        if firstUtp == firstBase:
            utp_inter += [iu]
#             print(firstUtp, firstBase)
        iu += 1
    
    utp_count = 0
    inter_final = []
    for firstUtp, secondUtp in zip(utp_inter[:-1], utp_inter[1:]):
        utp_period = secondUtp - firstUtp
        if utp_period > 180:
            if samp_utp[firstUtp:secondUtp].max() > utp_bplot[firstUtp:secondUtp].mean():
                print('({}, {})'.format(firstUtp, secondUtp))
                utp_count += 1
                inter_final += [firstUtp, secondUtp]

    return utp_count, utp_bplot, inter_final, samp_utp

def get_baseline(signal, winSize, slideInt, bymean=True):
    fhr_mod = signal    
    realB = []
    base_plot = np.zeros(len(fhr_mod))   # To plot the final baseline
    mTel = 0      # Counter
    while mTel <= len(base_plot): #-winSize:   ### Take only first 30 min of data 7200
        fhr_sub = fhr_mod[mTel:mTel + winSize]    ### get dhr values for size of window
        vir = fhr_sub.mean()   ### Calculate the virtual baseline
        f_h = vir + 8   ### max limit of BL (alpha = 8)
        f_l = vir -8    ### min limit of BL
        fhr_clamp = []
        ### Remove the accelerations and deaccelerations
        for f in fhr_sub:
            if f < f_l:
                fhr_clamp += [f_l]
            elif f > f_h:
                fhr_clamp += [f_h]
            else:
                fhr_clamp += [f]

        if bymean == True:
            real_b = int(np.array(fhr_clamp).mean())      ### Calculate real baseline by taking mode of clamped signal
            for fn, fh in enumerate(fhr_clamp):        ### error margin of +-5 for baseline
                if fh in range(real_b-5, real_b+5):
                    fhr_clamp[fn] = real_b
            real_count = np.where( np.array(fhr_clamp, dtype='int') == real_b )[0]
    #         if len(real_count) >= 480:       ### If signal for duration 2 min or greater, then accept otherwise reject
            realB += [real_b]
        else:
            real_b, real_count = stats.mode(fhr_clamp)    ### Calculate real baseline by taking mode of clamped signal
       
    #         if real_count >= 480:   ### If signal for duration 2 min or greater, then accept otherwise reject
    #             realB += [int(real_b)]
        
        base_plot[mTel:mTel + winSize] = int(real_b)
        mTel += slideInt   ### Update the slider according to slide_interval

    final_base = int(np.mean(np.array(realB)))  ### Take the mean of all baseline values to get final baseline

    return final_base, base_plot

def get_deacc(base, signal ):
    """
    base: [list or int], baseline of signal
    """
    
    if isinstance(base, int ):
        deacc_lim =  base #- 15     ### Add 15 bpm to baseline to get deacceleration limit
        fhr_intDec = list( map( lambda x: int(round(x)), signal ) )      ### save copy of fhr_mod with int values  
        fhr_intDec = np.array(fhr_intDec)
        deacc_plot = np.zeros_like(signal)
        deacc_plot[:] = deacc_lim
        for each_int in range(len(fhr_intDec)):        ### iterate over all values to round up values to deacc_lim for error window +-4
            try_b = fhr_intDec[each_int] 
            if try_b in range(deacc_lim-4, deacc_lim+4):
                fhr_intDec[each_int] = deacc_lim    ### include neighbouring points in threshold to get relaible intersection of signals
        
        inter_deacc = np.where(fhr_intDec == deacc_lim)[0]    ### Get interection points between new limit and fhr signal
        
        deacc_count = 0
        inter_final = []
        for firstDacc, currentDacc in zip( inter_deacc[:-1], inter_deacc[1:] ):
            deacc_period = currentDacc - firstDacc
            if deacc_period >= 60:            ###  Check if period is >= 15 seconds
                if fhr_intDec[ firstDacc : currentDacc ].min() < deacc_lim:
                    deacc_count += 1
                    inter_final += [firstDacc, currentDacc]
         
    else:
        base_plot = np.array(list( map( lambda x: int(round(x)), base ) ) )
        deacc_plot = base_plot - 15     ### Add 15 bpm to baseline to get deacceleration limit
        signal = list( map( lambda x: int(round(x)), signal ) )
        fhr_intDec = np.array(signal[:len(base_plot)])     ### save copy of fhr_mod with int values 
        for each_int in range(len(fhr_intDec)):        ### iterate over all values to round up values to deacc_lim for error window +-4
            try_b = fhr_intDec[each_int] 
            deacc_lim = deacc_plot[each_int]
            if try_b in range(deacc_lim-4, deacc_lim+4):
                fhr_intDec[each_int] = deacc_lim    ### include neighbouring points in threshold to get relaible intersection of signals

        inter_deacc = []    #### to store interscetions with baseline
        iu = 0
        for firstUtp, firstBase in zip(fhr_intDec, deacc_plot):
            if firstUtp == firstBase:
                inter_deacc += [iu]
    #             print(firstUtp, firstBase)
            iu += 1
    
        deacc_count = 0
        inter_final = []
        for firstDacc, currentDacc in zip( inter_deacc[:-1], inter_deacc[1:] ):
            deacc_period = currentDacc - firstDacc
            if deacc_period >= 60:            ###  Check if period is >= 15 seconds
                if (fhr_intDec[ firstDacc : currentDacc ] < deacc_plot[ firstDacc : currentDacc ]).any():
                    deacc_count += 1
                    inter_final += [firstDacc, currentDacc]
                    #print(firstDacc, currentDacc)

    return deacc_count, deacc_plot, inter_final, fhr_intDec

def get_accvar(base, signal):
    """
    base: [list or int] baseline of the signal
    """
    if isinstance(base, int ):
        pass
    else:
        base_plot = np.array(list( map( lambda x: int(round(x)), base ) ))
        signal = list( map( lambda x: int( round( float(x) ) ), signal ) )
        fhr_int = np.array(signal[:len(base_plot)])     ### save copy of fhr_mod with int values  
        for each_int in range(len(fhr_int)):        ### iterate over all values to round up values to baseline for error window +-3
            try_a = fhr_int[each_int]
            f_base = base_plot[each_int]
            if try_a in range(f_base-4,f_base+4):
                fhr_int[each_int] = f_base 
        
        fhr_intDec = np.array(signal[:len(base_plot)]) 
        acc_plot =  base_plot #+ 15     ### Add 15 bpm to baseline to get acceleration limit
        for each_int in range(len(fhr_intDec)):  ### use same copy of signal used for deacclereation
            try_b = fhr_intDec[each_int] 
            acc_lim = acc_plot[each_int]
            if try_b in range(acc_lim-4, acc_lim+4):
                fhr_intDec[each_int] = acc_lim    ### include neighbouring points in threshold to get relaible intersection of signals
        
        inter_acc = []    #### to store interscetions with baseline
        iu = 0
        for firstUtp, firstBase in zip(fhr_intDec, acc_plot):
            if firstUtp == firstBase:
                inter_acc += [iu]
    #             print(firstUtp, firstBase)
            iu += 1

        acc_count = 0
        temp_countint = []
        for firstacc, currentacc in zip( inter_acc[:-1], inter_acc[1:] ):
            acc_period = currentacc - firstacc
            if acc_period >= 60:            ###  Check if period is >= 15 seconds
                if fhr_intDec[ firstacc : currentacc ].max() > (acc_plot[ firstacc : currentacc ] + 15).mean() and ( fhr_intDec[ firstacc : currentacc ] >= acc_plot[ firstacc : currentacc ] ).all():
                    acc_count += 1
                    temp_countint += [firstacc, currentacc]
            
                    if acc_count == 1:   ### calculate variability 
                        var_start = np.where( fhr_int[currentacc:] == base_plot[currentacc:] )[0]   ### get the starting point of variability period
    #                     print(var_start)
                        if len(var_start) != 0:
                            var_start = currentacc + var_start[0]
                            var_end = var_start + (2 * 60 * 4)     ### variability period = 2 mins after first acceleration 

                            vYmin = fhr_int[var_start : var_end].min()    ### get Ymin and Ymax for the given period
                            vYmax = fhr_int[var_start : var_end].max()
                            fVariability = vYmax - vYmin       ###  Calculate variability 
                        else:
                            fVariability = 0
        if acc_count == 0:
            fVariability = 0
                        # temp_countint += [var_start]

    return acc_count, fVariability, acc_plot, temp_countint, fhr_intDec


def get_histfeat(fhr,):
    samp_fhr = np.array( list( map( lambda x: int(round(x)), fhr ) ) )
    val, bins = np.histogram(samp_fhr)
    #### calculate mean
    hist_mean = int(np.mean(samp_fhr))

    #### calculate median
    hist_median = int(np.median(samp_fhr))

    #### calculate mode
    mode, _ = stats.mode(samp_fhr)
    hist_mode = int(mode)

    #### calculate variance
    hist_variance = int(np.var(samp_fhr))
    
    #### calculate max and min and width
    hist_max = int(bins[-1])
    hist_min = int(bins[0])
    hist_width = int(bins[-1]-bins[0])
    
    return hist_mean, hist_median, hist_mode, hist_variance, hist_max, hist_min, hist_width

def get_heafeat(path):
    """ 
    path: (str) path to the .hea file
    """
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x.rstrip().lstrip() for x in lines]
    for line in lines:
        if line.startswith('#pH'):
            ph = float(line.split(' ')[-1])
            
        elif line.startswith('#pCO2'):
            pco = float(line.split(' ')[-1])
            
        elif line.startswith('#BE'):
            be = float(line.split(' ')[-1])
            
        elif line.startswith('#BDec'):
            bdecf = float(line.split(' ')[-1])
            
        elif line.startswith('#Gest'):
            gest = float(line.split(' ')[-1])
            
        elif line.startswith('#Age'):
            age = float(line.split(' ')[-1])
            
        elif line.startswith('#Weight'):
            weight = float(line.split(' ')[-1])
            
        elif line.startswith('#Gravidity'):
            grav = float(line.split(' ')[-1])
            
        elif line.startswith('#Parity'):
            parity = float(line.split(' ')[-1])

    return ph, pco, be, bdecf, gest, age, weight, grav, parity

def plot_signal(x_data, signal, inter=None, base=None, end_point=None, save_name=None):
    """
    inter: [list of int], list of the intersection points
    base: [list] baseline of the signal
    inter: [list] intersection points between signal and baseline
    end_point: [int] minutes to display
    save_name: [str] provide the name to save the plot
    """
    if end_point == None:
        i = len(signal)
    else:
        i = end_point * 240 
    

    plt.figure(figsize=(30,5))
    plt.plot(x_data[:i],signal[:i], '-',)
    if isinstance( base, (list, np.ndarray) ):
        base = list(base)
        plt.plot( x_data[:i], base[:i], '--' )

    if isinstance( inter, (list, np.ndarray) ):
        inter = list(inter)
        x_inter = [ x_data[ind] for ind in inter]
        y_inter = [ signal[ind] for ind in inter]
        plt.plot( x_inter, y_inter, '*', 'r' )

    plt.axis('on')
    plt.grid('on')
    plt.tight_layout()

    if save_name:
        plt.savefig(save_name, dpi=300, orientation='landscape')

    plt.show()

