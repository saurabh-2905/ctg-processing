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
from functions.utils import running_mean


def sis_utpcount(signal, winSize, slideInt,):
    """
    signal: raw utp signal without preprocessing
    winSize, slideInt: window size and slide interval (in terms of number of samples)
    """
    #### replace each value with avergage of 17 values
    utp = running_mean(signal, 17)

    utp_signal = list( map( lambda x: int(round(x)), utp ) ) 
    utp_signal = np.array(utp_signal)

    samp_utp = utp_signal[:7200]

    # utp_virbase = samp_utp.mean()
    utp_virbase, _ = stats.mode(samp_utp)

    utp_bplot = np.zeros(samp_utp.shape)   ### to store actual utp baseline
    for utpWin in range(0, 7200-winSize, slideInt):   ### get actual baseline for the give window size and slide interval
        #     print(utpWin)
        utp_mod = samp_utp[utpWin:utpWin+winSize]
        # utp_base = utp_mod.mean()
        utp_base, _ = stats.mode(utp_mod)
    #     print(utp_base)
        utp_bplot[utpWin:utpWin+winSize] = utp_base 

    utp_bplot = np.array(list( map( lambda x: int(round(x)), utp_bplot ) ))   ### get a smooth baseline curve for comparison

    utp_rplot = np.zeros(samp_utp.shape)   ### to store virtual utp baseline
    utp_rplot[:] = utp_virbase    #### curve for virtual baseline
    utp_rplot = np.array(list( map( lambda x: int(round(x)), utp_rplot ) )) 

    for each_int in range(len(samp_utp)):        ### iterate over all values to round up values to deacc_lim for error window +-4
        try_b = samp_utp[each_int] 
        deacc_lim = utp_bplot[each_int]  #utp_bplot[each_int]
        if try_b in range(deacc_lim-4, deacc_lim+4):
            samp_utp[each_int] = deacc_lim    ### include neighbouring points in threshold to get relaible intersection of signals
            #print(try_b)

    utp_count = 0
    inter_final = []
    mTel = 0
    while mTel !=2: 
        if mTel == 0:
            utp_inter = []    #### to store interscetions with baseline
            iu = 0
            for firstUtp, firstBase in zip(samp_utp[:len(utp_bplot)], utp_bplot):
                if firstUtp == firstBase:
                    utp_inter += [iu]
        #             print(firstUtp, firstBase)
                iu += 1
        for firstUtp, secondUtp in zip(utp_inter[:-1], utp_inter[1:]):
            utp_period = secondUtp - firstUtp
            mTel = 1
            if utp_period > 80 and utp_period < 960:
                if samp_utp[firstUtp:secondUtp].max() > utp_bplot[firstUtp:secondUtp].mean():
                    # print('({}, {}, {})'.format(firstUtp, secondUtp, utp_period))
                    utp_count += 1
                    inter_final += [firstUtp, secondUtp]
                    end_utp = secondUtp
            elif utp_period >=960:
                utp_re = samp_utp[firstUtp:secondUtp]
                utp_re_base, _ = stats.mode(utp_re)
                utp_bplot[firstUtp:secondUtp] = utp_re_base
                mTel = 0
                break
        if secondUtp == utp_inter[-1]:
            mTel = 2

    return utp_count, utp_bplot, inter_final, samp_utp

def sis_astv(signal):
    """
    signal: raw/pre-processed fhr signal

    
    """
    signal = list(signal)
    signal =  list( map( lambda x: int(round( float(x) )), signal ) )
    signal = np.array(signal)

    val_pix = []
    for i, (firstpt, secondpt) in enumerate(zip(signal[:-1], signal[1:])):
        diffpt = abs(firstpt-secondpt)
        if diffpt < 1:
            val_pix +=[firstpt]

    astv = round( (len(val_pix) / len (signal)) * 100)
    return val_pix, astv

def sis_accshift(base, signal, astv):
    """
    base: [list or int], baseline of signal
    """
    if isinstance(base, int ):
        pass
    else:
        base_plot = np.array(list( map( lambda x: int(round(x)), base ) ))
        signal = list( map( lambda x: int( round( float(x) ) ), signal ) )
        fhr_intDec = np.array(signal[:len(base_plot)]) 
        acc_plot =  base_plot #+ 15     ### Add 15 bpm to baseline to get acceleration limit
        for each_int in range(len(fhr_intDec)):  ### use same copy of signal used for deacclereation
            try_b = fhr_intDec[each_int] 
            acc_lim = acc_plot[each_int]
            if try_b in range(acc_lim-4, acc_lim+4):
                fhr_intDec[each_int] = acc_lim    ### include neighbouring points in threshold to get relaible intersection of signals

        mTel = 0
        while mTel != 1:
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
                if acc_period >= 60 and acc_period < 480:            ###  Check if period is >= 15 seconds
                    if fhr_intDec[ firstacc : currentacc ].max() > (acc_plot[ firstacc : currentacc ] + 15).mean() and ( fhr_intDec[ firstacc : currentacc ] >= acc_plot[ firstacc : currentacc ] ).all():
                        acc_count += 1
                        temp_countint += [firstacc, currentacc]
                elif acc_period >= 480:
                    up_bl = sis_baseline(fhr_intDec[ firstacc : currentacc ], astv)
                    acc_plot[ firstacc : currentacc ] = up_bl[0]
                    break

            if currentacc == inter_acc[-1]:
                mTel = 1

    return acc_count, acc_plot, temp_countint, fhr_intDec   


def sis_deacc(base, signal):
    """
    base: [list or int], baseline of signal
    """

    if isinstance(base, int ):
        deacc_lim =  base #- 15     ### Add 15 bpm to baseline to get deacceleration limit
        fhr_intDec = np.array(list( map( lambda x: int(round(x)), signal ) ))      ### save copy of fhr_mod with int values  
        deacc_plot = np.zeros_like(signal)
        deacc_plot[:] = deacc_lim
        for each_int in range(len(fhr_intDec)):        ### iterate over all values to round up values to deacc_lim for error window +-4
            try_b = fhr_intDec[each_int] 
            if try_b in range(deacc_lim-4, deacc_lim+4):
                fhr_intDec[each_int] = deacc_lim    ### include neighbouring points in threshold to get relaible intersection of signals
        
        inter_deacc = np.where(fhr_intDec == deacc_lim)[0]    ### Get interection points between new limit and fhr signal
        
        light_count = 0
        severe_count = 0
        prolonged_count = 0
        inter_final = []
        for firstDacc, currentDacc in zip( inter_deacc[:-1], inter_deacc[1:] ):
            deacc_period = currentDacc - firstDacc
            if fhr_intDec[ firstDacc : currentDacc ].min() < deacc_lim:
                if deacc_period >= 60 and deacc_period <= 480:            ###  Check if period is >= 15 seconds
                    light_count += 1
                    inter_final += [firstDacc, currentDacc]
                elif deacc_period > 480 and deacc_period <= 1200: 
                    severe_count += 1
                    inter_final += [firstDacc, currentDacc]
                elif deacc_period > 1200:
                    prolonged_count += 1
                    inter_final += [firstDacc, currentDacc]

    else:
        base_plot = np.array(list( map( lambda x: int(round(x)), base ) )) 
        deacc_plot = base_plot    #- 15     ### Add 15 bpm to baseline to get deacceleration limit
        signal = np.array(list( map( lambda x: int(round( float(x) )), signal ) ))
        fhr_intDec = signal[:len(base_plot)]     ### save copy of fhr_mod with int values 
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
    
        light_count = 0
        severe_count = 0
        prolonged_count = 0
        inter_final = []
        for firstDacc, currentDacc in zip( inter_deacc[:-1], inter_deacc[1:] ):
            deacc_period = currentDacc - firstDacc
            if (fhr_intDec[ firstDacc : currentDacc ] <= deacc_plot[ firstDacc : currentDacc ]).all() and fhr_intDec[ firstDacc : currentDacc ].min() < (deacc_plot[ firstDacc : currentDacc ] - 15).mean() :
                if deacc_period >= 60 and deacc_period <= 480:            ###  Check if period is >= 15 seconds
                    light_count += 1
                    inter_final += [firstDacc, currentDacc]
                elif deacc_period > 480 and deacc_period <= 1200: 
                    severe_count += 1
                    inter_final += [firstDacc, currentDacc]
                elif deacc_period > 1200:
                    prolonged_count += 1
                    inter_final += [firstDacc, currentDacc]

    return [light_count, severe_count, prolonged_count], deacc_plot, inter_final, fhr_intDec

def sis_altv( signal, x, acc_inter, decc_inter):   #base,
    """
    base: [list or int], baseline of signal
    """
    # signal = np.array(list( map( lambda x: int(round(x)), signal[:len(base)] ) )) 
    # vir_base = np.mean(signal)
    # upper_lim, lower_lim = base+15, base-15

    # sig_mod = []
    # og_ind = []
    # for i, sig in enumerate(signal):
    #     if sig < upper_lim[i] and sig > lower_lim[i]:
    #         sig_mod += [sig] 
    #         og_ind += [i]                     #### consider the signal not used for acceleration and deacc
    
    # winSize = 240
    # slideInt = 40
    # altv_points = []
    # altv_count = 0
    # mTel = 0
    # while mTel < len(sig_mod) - winSize:         #### check for altv over window of size 60 sec
    #     ltv_check = sig_mod[mTel:mTel+winSize]
    #     ltv_bpm = np.array(ltv_check).max() - np.array(ltv_check).min()
    #     ### if condition satisfies store the index of center point w.r.t original signal
    #     if ltv_bpm < 5:
    #         altv_points += [ og_ind[mTel+120] ]
    #         altv_count += 1
        
    #     mTel += slideInt

    # altv = round( (altv_count / len(signal)) * 100 )
    
    signal = np.array(list( map( lambda x: int(round(x)), signal ) )) 

    total_inter = sorted(acc_inter + decc_inter)
    total_inter = [0] + total_inter + [len(signal)]

    sig_mod = []
    og_ind = []
    for fir, sec in zip( total_inter[::2], total_inter[1::2] ):
        sig_mod.extend(signal[fir:sec]) 
        ind = list( np.linspace( fir, sec, len(signal[fir:sec])+1, dtype='int' ) )  
        og_ind = og_ind + ind

    winSize = 240
    slideInt = 1
    altv_points = []
    altv_count = 0
    mTel = 0
    while mTel < len(sig_mod) - winSize :         #### check for altv over window of size 60 sec
        ltv_check = sig_mod[mTel:mTel+winSize]
        ltv_bpm = np.array(ltv_check).max() - np.array(ltv_check).min()
        ### if condition satisfies store the index of center point w.r.t original signal
        if ltv_bpm < 5:
            altv_points += [ og_ind[mTel+120] ]
            altv_count += 1
        
        mTel += slideInt

    altv = round( (altv_count / len(signal)) * 100 )
    
    return altv_points, altv
    
def select_signal(fhrf, utpf, x_dataf, ):
    zero_ind = np.where(np.array(fhrf) != 0)[0]   # storing index for non_zeros instead

    ##### check for the continuity of segment and store the segments
    each_seg = []
    seg_chunks = []
    for fi, se in zip(zero_ind[:-1], zero_ind[1:]):
        period = se-fi
        if period == 1:
            each_seg += [fi]
        elif period > 1:
            if period > 60:   # check for gap greater than 15 sec
                each_seg += [fi]
                seg_chunks +=[each_seg]
                each_seg = []
    #             print(period)
    #             break
            else:
                each_seg += [fi]
    seg_chunks += [each_seg]

    len_seg = []
    for each in seg_chunks:
        len_seg += [ len(each) ]
        # print(len(each))
        
    long_ind = int(np.where( len_seg == np.array(len_seg).max() )[0])

    fhr_mod = fhrf[ seg_chunks[long_ind] ]
    utp_mod = utpf[ seg_chunks[long_ind] ]
    x_mod = x_dataf[ seg_chunks[long_ind] ]
    start_time = x_mod[0]
    end_time = x_mod[-1]

    return fhr_mod, utp_mod, x_mod, start_time, end_time
    
def sis_baseline(fhr_mod, astv):
    fhr_mod1 = np.array(list( map( lambda x: int(round(x)), fhr_mod ) ) )  #### convert floating values to int
    unique_val = np.unique(fhr_mod1)    ### extract the unique values
    base_plot = np.zeros(len(fhr_mod))   # To plot the final baseline

    ###### get frequency of occurence for each unique value
    hifi = []
    for un in unique_val:
        hifi += [(len(np.where(fhr_mod1 == un)[0]),un)]
    #     break

    ####### arrange the value in descending order of frequency and check if total occurance is greater than 5% of total signal
    hifi.sort(key=lambda x: x[0])
    hifi = hifi[::-1]
    hifi_50 = list(filter(lambda x: x[0] > round(fhr_mod1.shape[0] * 0.05), hifi))
    
    ####### if no values occur for 5% of the total signal, consider all other values
    if len(hifi_50) == 0:
        hifi_50 = hifi
        print(hifi,  round(fhr_mod1.shape[0] * 0.05))
    
    if len(hifi_50) > 50:
        hifi_50 = hifi[:50] 

    ##### baseline is the value with highest frequency    
    bl = hifi_50[0][1]
    H1 = hifi_50[0][0]
    f_factor = 0
    set_break = 0

    ##### check if the baseline is above 110 bpm
    if bl >= 110: 
        if bl > 152:
            ####### check if any other fi 110 <= fi < bl and hi > 1.6 * astv * h1
            for it1 in hifi_50[1:]:
                if it1[1] >= 110 and it1[1] < bl and it1[0] > 1.6*astv*H1:
                    if it1[1] < bl:
                        bl = it1[1]
                    
        else:   ###### bl < 152
            if astv < 20:
                f_factor = 4
            elif astv >=20 and astv <30:
                f_factor = 2
            elif astv >=30 and astv <40:
                f_factor = 1 
            elif astv >=40 and astv <60:
                f_factor = 0.5
            elif astv >=60:
                f_factor = 1
            ####### check if any other fi 110 <= fi < bl and hi > F * astv * h1    
            for it1 in hifi_50[1:]:
                if it1[1] >=110 and it1[1] < bl and it1[0] > f_factor*astv*H1:
                    if it1[1] < bl:
                        bl = it1[1]
        
    else:   ##### bl < 110
        ######## check if any other fi > 110 and hi > (100-astv)/3*H1
        for it1 in hifi_50[1:]:
            if it1[1] > 110 and it1[0] > (100-astv)/3 * H1:
                bl =it1[1]
                set_break = 1
                break     ####### stop the iteration of baseline achieved
        
        ####### only check this condition if baseline not obtained by previous check
        if set_break == 1:
            pass
        else:
            ######### check if any other fi < bl and hi >astv*h1
            for it1 in hifi_50[1:]:
                if it1[1] < bl and it1[0] > astv*H1:
                    if it1[1] < bl:
                        bl = it1[1]
    base_plot[:] = bl
    
    return bl, base_plot

def sis_rmspike(signal, x_data):
    signal = np.array(list( map( lambda x: int(round(x[0])), signal ) ))

    qTel = 0
    while qTel < len(signal)-1:
        first_sig = signal[qTel]
        second_sig = signal[qTel+1]

        diff_sig = abs(first_sig - second_sig)

        if diff_sig > 25:
            signal[qTel+1] = 0
            j = 2
            mTel = 0
            while mTel != 1:
                diff_sig = abs(first_sig - signal[qTel+j])
                if diff_sig > 25:
                    signal[qTel+j] = 0
                    j += 1
                else:
                    diff_1 = abs(signal[qTel+j] - signal[qTel+j+1])
                    diff_2 = abs(signal[qTel+j+1] - signal[qTel+j+2])
                    diff_3 =  abs(signal[qTel+j+2] - signal[qTel+j+3])
                    diff_4 =  abs(signal[qTel+j+3] - signal[qTel+j+4])
                    if diff_1 < 10 and diff_2 <10 and diff_3 <10 and diff_4 <10:
                        mTel = 1
                        qTel = qTel + j + 4
                    else:
                        j += 1
        else:
            qTel +=1
    ###### once the spikey points replaced with zero perform interpolation to fill holes
    nz_ind = np.where(signal != 0)[0]

    tointer_sig = signal[nz_ind]
    tointer_x = x_data[nz_ind]

    sig_ob = interp1d(tointer_x, tointer_sig)

     #### introduce upper and lower bound for x data
    filtered_ind = np.where(np.array(x_data) > tointer_x[0])[0]
    x = x_data[filtered_ind]
    #x = [ x[ind] for ind in filtered_ind ]
    filtered_ind = np.where(np.array(x) < tointer_x[-1])[0]
    # x = [ x[ind] for ind in filtered_ind ]
    x = x_data[filtered_ind]

    signal = [ sig_ob( ind ) for ind in x ]
    

    assert len(signal) == len(x)

    return signal, x