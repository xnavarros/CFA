# ------------------------------------------------------------------------------------------------------------------------------
# Copyright (C) 2024 Xavier Navarro-Suné - All Rights Reserved
#
# You may use, distribute and modify this code under the terms of the GNU GPL license. 
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. 
# If not, see <http://www.gnu.org/licenses/>
# ------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import mne
import warnings
from scipy.signal import decimate, resample_poly
from mne.stats import permutation_cluster_test
from mne.time_frequency import tfr_morlet, tfr_array_morlet
from skimage import measure

def time_to_phase_average(tfr_pwr, times, phase_step, ini_phase, end_phase, verbose):
    """
    Convert Time-Frequency maps to Cycle-Frequency maps
    @params:
        tfr_pwr     - Required  :  TFR data in numpy 4D array(N_trials, N_channels, N_frequencies, N_times)
        times       - Required  :  1D numpy array containing the original times 
        phase_step  - Required  :  Phase step determining the phase resolution of Cycle-frequency maps (1 recommended)
        ini_phase   - Required  :  Initial value of the phase with respect to the cycle (2*pi) 
        end_phase   - Required  :  End value of the phase with respect to the cycle (2*pi) 
        verbose     - Required  :  Display progress bar and other information
    
    """
    # check input tfr_pwr size
    if tfr_pwr.ndim < 4:
        print("* Error: tfr_pwr must be a 4D array")
        exit()
    
    # check if phase_step is integer
    if not (isinstance(phase_step, int)):
        print("* Error: Noninteger phase step")
        exit()
    
    # check if ini_phase and end_phase are correct
    if ini_phase > 0:
        print("* Error: Initial phase must be negative")
        exit()
    
    # Compute phase dimension and bin  
    phase_span = (abs(ini_phase)+abs(end_phase))*360
    nb_bins_deg = times.size / phase_span 
    nb_bins_step = round(nb_bins_deg*phase_step)
    
    # Check if dim_phase is not integer
    dim_phase = phase_span / phase_step
    if not dim_phase.is_integer(): 
        warnings.warn("The phase span (in degrees) must be divisible by the phase step.")
        print("New dimension : ",  int(dim_phase))  
        
    # initialise new phase-frequency maps
    # tfr_pwr = (N_segments, N_channels, N_freqs, N_times)
    pfr_pwr = np.zeros((tfr_pwr.shape[0],tfr_pwr.shape[1],tfr_pwr.shape[2],int(dim_phase)),dtype=complex)
    up_factor = int(dim_phase)
    down_factor = tfr_pwr.shape[3]
    if verbose:
        print("TF map dimension : ", tfr_pwr.shape)
        print("Cycle-frequency maps dimension :", pfr_pwr.shape)
        print("Nb of bins per degree= ", nb_bins_deg, "; #bins per step of ", phase_step, " deg :",nb_bins_deg*phase_step) 
        print("Resampling factors: up=", up_factor, " ; down=",down_factor)
        print("Resampling using resample_poly function")
       
    
    # create phases array
    phases = np.arange(int(ini_phase*360), int(end_phase*360), int(phase_step))
    l = tfr_pwr.shape[0]*tfr_pwr.shape[1]
    if verbose:
        printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 70)
        count=0
    for ch in range(tfr_pwr.shape[1]): # loop channels
        for x in range(tfr_pwr.shape[0]): # loop trials    
            #pfr_pwr[x,ch,:,:] = resize(tfr_pwr[x,ch,:,:], (pfr_pwr.shape[2], pfr_pwr.shape[3]),anti_aliasing=True)         
            #pfr_pwr[x,ch,:,:] = decimate(tfr_pwr[x,ch,:,:], resamp_factor, ftype='fir',axis=1) 
            pfr_pwr[x,ch,:,:] = resample_poly(tfr_pwr[x,ch,:,:], up_factor, down_factor, axis=1)

            # update progress bar
            if verbose:
                count = count+1
                printProgressBar(count, l, prefix = 'Progress:', suffix = 'Complete', length = 70) 
    
    return pfr_pwr, phases


def time_to_phase_by_cycle(data, sfreq, triggers, ph_step, ini_phase, end_phase, freqs, n_cycles, min_ttot, max_ttot, segmentation, verbose):
    """
    Convert continuous data in segments adapted to inter-trigger times
    @params:
        data          - Required  :  Unsegmented EEG data in numpy 2D array [N_chans, N_times]
        sfreq         - Required  :  Sampling frequency in Hz
        triggers      - Required  :  1D array containing the trigger instants (nb of samples) 
        ph_step       - Required  :  Phase step determining the phase resolution of Cycle-frequency maps (1 recommended)
        ini_phase     - Required  :  Initial value of the phase with respect to the cycle (2*pi) 
        end_phase     - Required  :  End value of the phase with respect to the cycle (2*pi) 
        freqs         - Required  :  1D array containing the frequencies to run TFR
        n_cycles      - Required  :  Number of cycles to perform TFR 
        min_ttot      - Required  :  Minimal duration of a cycle to perform TFR
        max_ttot      - Required  :  Maximal duration used as threshold duration if current cycle exceeds max_ttot
        segmentation  - Required  :  - [ini, end]: fixed segmentation using given limits in seconds; 
                                     - 0: Cycle length adapted segmentation 
        verbose     - Required  :  Display progress bar and other information
    
    """
    if verbose:
        if segmentation==0:
            print("Splitting continuous data into segments of length adapted to each cycle duration")
        else:
            if len(segmentation)==2:
                print("Splitting continuous data into segments of fixed length:", segmentation)
                print("  Processing", int(triggers.size), "epochs")
            else:
                print("Error, incorrect segmentation limits")
    
    if verbose:
        printProgressBar(0, triggers.size-1, prefix = 'Progress:', suffix = 'Complete', length = 70)
        count=0    
    
    n_epochs = int(triggers.size-1)
    if data.ndim == 1: 
        n_chans = 1
        data = np.reshape(data, (1,data.shape[0]))
    else : n_chans  = data.shape[0]   
       
    n_phase_bins = int(((-ini_phase+end_phase)*360)/ ph_step)
    pfr_pwr = np.empty((n_epochs, n_chans ,freqs.size,n_phase_bins),dtype=complex)
    pfr_pwr[:] = np.nan
    
    
    for i in range(1,triggers.size):
        nb_samp_cycle = triggers[i] - triggers[i-1]
        ttot = nb_samp_cycle/sfreq
        if ttot < min_ttot:
            # skip this trial
            warnings.warn(message = "The duration of the cycle is shorter than the minimal duration min_ttot")
            print("Cycle number:",i ," triggers[i]=",triggers[i],"triggers[i-1]=",triggers[i-1])
            print("ttot=",ttot, "(min_ttot=", min_ttot, ")")
        else:
            if segmentation==0:
                ini = int(ini_phase*nb_samp_cycle + triggers[i])
                end = int(end_phase*nb_samp_cycle + triggers[i])
                time = np.arange(ttot*ini_phase,ttot*end_phase,1/sfreq)
                if (end-ini)/sfreq > max_ttot:
                    warnings.warn(message = "The duration of the cycle is longer than the maximal duration max_ttot")
                    print("Cycle number:",i ," triggers[i]=",triggers[i],"triggers[i-1]=",triggers[i-1])
                    end = ini+int(max_ttot*sfreq)
                    print("ini=",ini, "end", end, " final duration=", (end-ini)/sfreq)                   
            else:
                ini = int(segmentation[0]*sfreq + triggers[i])
                end = int(segmentation[1]*sfreq + triggers[i])
                time = 1e3 * np.arange(ini,end,1/sfreq)
                
            #print("i=",i,"Ini sample:", ini, " end sample", end, " = ", (end-ini)/sfreq, "sec ; ") 
            epoch = data[:,ini:end]
           
            if time.size > epoch.shape[1]:
                time = time[0:epoch.size]
                #print("correcting time length")
            if time.size < epoch.shape[1]:
                epoch = epoch[:,0:time.size]
                #print("correcting epoch length")
        
            data_epoch = np.reshape(epoch, (1,n_chans,epoch.shape[1]))
            # compute tfr
            tfr_epoch = tfr_array_morlet(data_epoch, sfreq, freqs, n_cycles=n_cycles, output='complex')
            # Convert to phase 
            pfr_pwr[i-1,:,:,:],phases = time_to_phase_average(tfr_epoch,time,ph_step, ini_phase, end_phase, False)
            if verbose:
                count = count+1
                printProgressBar(i, triggers.size-1, prefix = 'Progress:', suffix = 'Complete', length = 70)
    
    return pfr_pwr,phases


def permutation_test_plot(epochs_power_1, epochs_power_2, xvals, freqs, threshold, n_permutations, left_exclude, right_exclude, alpha, contours):
    """
    Perform permutation test using either two conditions or one condition for TF or CF maps. 
    If one condition, a baseline period must be specified in epochs_power_2
    @params:
        epochs_power_1       - Required  :  4D array containing TF/CF maps for 1st condition [N_trials1, N_chan, N_freqs, N_x]
        epochs_power_2       - Required  :  4D array containing TF/CF maps for 2st condition [N_trials2, N_chan, N_freqs, N_x], or
                                         :  Baseline limits, 1D array containing (ind1, ind2) 
        xvals                - Required  :  1D array containing values of x axis (time or phase) 
        freqs                - Required  :  1D array containing frequency values
        threshold            - Required  :  Significance threshold used in permutation tests, see permutation_cluster_test
        n_permutations       - Required  :  Number of permutations for permutation_cluster_test function
        left_exclude         - Required  :  Number of leftmost samples to be excluded to avoid edge effects
        right_exclude        - Required  :  Number of rightmost samples to be excluded to avoid edge effects
        alpha                - Required  :  Alpha level for statistical test
        contours             - Required  :  Compute or not contours of singinficant areas
    """
    if epochs_power_2.size == 2:
        # generate epochs_power_2 replicating baseline content from epochs_power_1
        #print("Generating comparison condition from baseline")
        i1 = epochs_power_2[0]
        i2 = epochs_power_2[1]
        epochs_power_2 = replicate_baseline_CF_map(epochs_power_1,i1,i2)      
    
    exclude = np.zeros((epochs_power_1.shape[1:]), dtype=bool)
    exclude[:,-right_exclude:] = True
    exclude[:,0:left_exclude] = True

    F_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test([epochs_power_1, epochs_power_2], out_type='mask', n_permutations=n_permutations, 
                                 threshold=threshold, exclude=exclude, tail=0)

    # Compute the difference in evoked to determine which was greater since
    # we used a 1-way ANOVA which tested for a difference in population means
    evoked_power_1 = epochs_power_1.mean(axis=0)
    evoked_power_2 = epochs_power_2.mean(axis=0)
    evoked_power_contrast = evoked_power_1 - evoked_power_2
    signs = np.sign(evoked_power_contrast)

    # Create new stats image with only significant clusters
    F_obs_plot = np.nan * np.ones_like(F_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= alpha:
            F_obs_plot[c] = F_obs[c] * signs[c]   
    
    if contours:
        # Find contours of significant areas to supperpose to power maps 
        F_contour = (F_obs_plot*0)+1
        F_contour[np.isnan(F_contour)] = 0
        cont = measure.find_contours(F_contour,level=0, fully_connected='low') 
    else:
        cont = 0
        
    return F_obs, F_obs_plot, cont    


def apply_baseline_CF_map(cfr,ind1,ind2,relative):
    """
    Baseline correction on trials by substracts baseline or computing "percent" baseline (changes in % with respect to baseline)
    @params: 
            cfr          - Required  : CFR map [N_trials, [N_chan], N_freqs, N_phases]
            ind1         - Required  : index to first position of baseline, typically 0
            ind2         - Required  : index to baseline's end position
            relative     - Required  : 0: baseline subtraction ; 1: relative changes
    """
    cfr_bl = np.empty((cfr.shape))
    
    try:
        assert(cfr.ndim >= 3) 
        if cfr.ndim == 3:
            for i in range(cfr.shape[0]):
                A = cfr[i,:,ind1:ind2]
                # skip if it's an empty slice 
                if np.nansum(A)>0:
                    a=np.nanmean(A,axis=1)
                    b=a[np.newaxis]
                    cfr_bl[i,:,:] = cfr[i,:,:] - b.T
                    if relative:
                        cfr_bl[i,:,:] = cfr_bl[i,:,:]/b.T
        else:
            for j in range(cfr.shape[1]):
                 for i in range(cfr.shape[0]):
                    A = cfr[i,j,:,ind1:ind2]
                    # skip if it's an empty slice 
                    if np.nansum(A)>0:
                        a=np.nanmean(A,axis=1)
                        b=a[np.newaxis]
                        cfr_bl[i,j,:,:] = cfr[i,j,:,:] - b.T
                        if relative:
                            cfr_bl[i,j,:,:] = cfr_bl[i,j,:,:]/b.T
        return cfr_bl
    except:
        print("Baseline error: Matrix must be 3D or 4D")
    
    
def replicate_baseline_CF_map(cfr,i1,i2):
    """
    Constructs a set of CF maps of identical size than pfr, but using only data from the specified intervals (baseline) 
    @params: 
            cfr          - Required  : 3D or 4D PFR map [N_trials, [N_chan], N_freqs, N_phases]
            ind1         - Required  : index to first position of baseline, typically 0
            ind2         - Required  : index to baseline's end position     
    """
    di = i2-i1
    side_eff_corr = 5
    try:
        assert(di >= 1)
        if cfr.ndim==4:    
            bl = cfr[:,:,:,i1:i2]
            factor = np.ceil(cfr.shape[3]/bl.shape[3])
            rep_bl = bl.repeat(factor,axis=3)
            rep_bl = rep_bl[:,:,:,:cfr.shape[3]]
        if cfr.ndim==3:
            bl = cfr[:,:,i1:i2]
            factor = np.ceil(cfr.shape[2]/bl.shape[2])
            rep_bl = bl.repeat(factor,axis=2)
            rep_bl = rep_bl[:,:,:cfr.shape[2]]        
        
        return rep_bl
    except:
        print("* Error: Incorrect baseline indices")
    

def compute_itc_pfr(cfr):
    """
    Compute inter-trial coherence in cycle-frequency maps
    cfr : complex-valued 4D array [N_trials, N_chan, N_freqs, N_phases]
    """
    assert(cfr.ndim == 4)
    
    # do loops accross trials and channels to skip nans
    itc = np.empty(cfr.shape[1:])
    for j in range(cfr.shape[1]):
        itt = np.zeros(cfr.shape[2:],dtype=complex)
        N = 0
        for i in range(cfr.shape[0]):
            cfrc = cfr[i,j,:,:]
            if np.nansum(cfrc)>0:
                N=N+1
                itt = itt + cfrc/np.abs(cfrc)
        itc[j,:,:] = np.abs(itt)/N  
    return itc 
    
    
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

        