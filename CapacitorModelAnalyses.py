#######################################################
# Capacitor Model Calculations:                       #
# -----------------------------                       #
#                                                     #
# Run capacitor model for flash data provided.        #
#    -) Runaway electric field breakeven threshold    #
#       calculation for initiation altitudes.         #
#    -) Critical surface and volume charge density    #
#    -) retrievals.                                   #
#    -) Capacitor energy calculation for evaluation   #
#       agains the provided data, and exploration     #
#       of its dependancies.                          #
#                                                     #
# Run Time Series Analyses:                           #
#  -Provides time series binning and plotting for all #
#   figures shown in the manuscript.                  #
#                                                     #
# Run Figure plotting to reproduce figures in         #
# the manuscript - violin plots, etc.                 #
#######################################################

import numpy as np
import pandas as pd
import scipy.stats as st
import Capacitor as cap
from EnergyPlotting import hist_min,box_bins,energy_time_series,energy_compare,box_plots,eta_time_series,cov_time_series,energy_hists
from CasePlotting import case_time_series, case_energy_area
from Capacitor import e_br, compute_energy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_orig()


#################################################
#Specify Case Provided in Datasets: (WK82, SL16)#
#################################################
case_wk     = 'WK'
case_sl     = 'SL'
bin_range   = 60.   #for one minute bins - can change for other desired bin spacings
uniform_eta = False # If uniform_eta is True, specify the eta_c values to be used to adjust energy values.
eta_u       = 0.008 # uniform eta value (u = uniform) - median range between 0.008-0.01


####################################################
# Cell Splitting Routines to extract cell-specific #
# flash populations.                               #
# note: for partioned model                        #
####################################################
def flash_filter(case):
    cg_flags  = pd.read_csv(f'InitData/{case}-CG-Flashes.csv')
    cg_ignore = cg_flags.cg_flag.values
    return(cg_ignore)

def data(case):
    cg_ignore = flash_filter(case)
    data = f'InitData/{case}_INIT_DATA.csv'
    data = pd.read_csv(data)
    data = data.drop(index=cg_ignore)

    if case == 'WK':
        cname = 'WK82'
    else:
        cname = 'SL16'
    datq = f'InitData/{cname}_INITS_Analysis_FINAL-NEW.csv'
    datq = pd.read_csv(datq)
    datq = datq.drop(index=cg_ignore)
    return(data,datq)

def cell_subset(case,data,datq,cell):
    if case == 'WK':
        split_axis = 29e3
    elif case == 'SL':
        split_axis = 50e3
    else:
        raise ValueError('Invalid Case Chosen, options include WK and SL.')

    y_dim     = data.yi*125.
    cell_dict = {'right':(y_dim<split_axis),'left':(y_dim>split_axis),'full':(y_dim>0)}
    if cell in cell_dict.keys():
        cell_mask = cell_dict[cell]

    data = data.iloc[np.array(cell_mask)]
    datq = datq.iloc[np.array(cell_mask)]
    return(data,datq,cell_mask)


def get_vars(subset,lo,ln):
    itime           = subset['Time (s)']
    einit           = e_br((subset.zi*125.)*1e-3)*1e3
    dat_area        = subset.area
    dat_sep         = subset.separation
    cap_eng = compute_energy(dat_sep,dat_area,einit,'surface')
    com_eng         = np.abs(subset[' Change in Energy'])
    com_eta         = com_eng/cap_eng
    eta_m           = com_eng/subset[' Total Electrical Energy']
    cap_adjust      = cap_eng.copy() * np.nanmedian(com_eta)
    var_dict        = {'einit':einit,'area':dat_area,'sep':dat_sep,
                       'cap':cap_eng,'com_eng': com_eng,'etac':com_eta,
                       'etam':eta_m,'cap_adj':cap_adjust,'itime':itime,
                       'lo':lo,'ln':ln}

    return(var_dict)

def get_lengths(subset):
    meanL = np.mean(subset['area']**0.5)
    std   = np.std( subset['area']**0.5) * .45#.45
    l_len = meanL-std
    r_len = meanL+std
    lo = np.log10(l_len)
    ln = np.log10(r_len)
    return(l_len,r_len,lo,ln)

######################################################
#Read Data: And set some global variables to be used #
######################################################
def get_flash_data(case,charge_type):
    '''
    Read data file with initiation data. Returns tuple of:
        read_flash   = case data frame
        model_energy = COMMAS flash energy change
        init_time    = flash initiation time
        init_altitude= flash initiation altitude
        plate_separation= charge center spacing
        flash_breakdown = critical electric field value
        plate_area      = flash area
        cap_energy      = modeled capacitor flash energy change
    '''
    read_flash   = pd.read_csv(f'InitData/{case}_INIT_DATA.csv')
    cg_flashes   = pd.read_csv(f'InitData/{case}-CG-Flashes.csv')
    ignore = cg_flashes.cg_flag.values

    # read_flash   = read_flash.convert_objects(convert_numeric=True)
    read_flash   = read_flash.apply(pd.to_numeric, errors="ignore")
    read_flash   = read_flash.dropna()
    read_flash   = read_flash.drop(index=ignore) #Drop all flashes that correspond to -/+CG or IC flashes with a CG component

    if case == 'SL':
        mask = np.where(read_flash.yi*125 < 50e3)
        read_flash = read_flash.iloc[mask]

    model_energy = -read_flash[' Change in Energy'] #COMMAS Flash energy change (- sign as values are energy removed)
    init_time    = read_flash['Time (s)']         #Time of initiation


    #Compute Breakdown Field
    init_altitude       = read_flash.zi *125    #zi is given as index location - convert by multiplying by grid spacing (m)
    plate_separation    = read_flash.separation # meters
    flash_breakdown     = cap.e_br(np.abs(init_altitude)*1e-3)*1e3 #V/m
    #Now define capacitor geometrical values:
    plate_area          = read_flash.area*1e6   #cubic meters | m^3

    ####################################
    #Compute flash capacitor energy:   #
    ####################################
    cap_energy  = cap.compute_energy(plate_separation,plate_area,flash_breakdown,charge_type) #Joules (J)

    minute_bins = np.array(np.arange(init_time.min(), init_time.max(),bin_range))

    return(read_flash,model_energy,init_time,init_altitude,
           plate_separation,flash_breakdown,plate_area,cap_energy,minute_bins)


def data_tuples(wk,sl):
    '''
    Get tuples of binned data: tedges,commas_totals and means, cap_totals and means
    '''
    wk_tedges,wk_commas_totals,wk_commas_emans,wk_cap_totals,wk_cap_means = wk
    sl_tedges,sl_commas_totals,sl_commas_emans,sl_cap_totals,sl_cap_means = sl

    t_centers     = (wk_tedges,sl_tedges)
    commas_totals = (wk_commas_totals,sl_commas_totals)
    commas_means  = (wk_commas_means ,sl_commas_means )
    cap_totals    = (wk_cap_totals   ,sl_cap_totals   )
    cap_means     = (wk_cap_means    ,sl_cap_means    )
    return(t_centers,commas_totals,commas_means,cap_totals,cap_means)


####################################
#Data Bins for histograms:         #
####################################

if __name__ == '__main__':
    #Data for Main Plots:
    wk_data = get_flash_data(case_wk,'surface') #charge_type == 'surface' implies surface charge density calculation for W_c; else use space charge density.
    sl_data = get_flash_data(case_sl,'surface') #charge_type == 'surface' implies surface charge density calculation for W_c; else use space charge density.

    #Temporary Work around, but open data to generate dictionary objects for full storm cells:------
    #Get variables for plotting cell specific flashes (i.e., left, right, or full storm):
    case_1 = 'WK'
    case_2 = 'SL'

    wk,wkq = data(case_1) #wk, and wkq are different dataframes; wk includes locations only, wkq contains .output data.
    sl,slq = data(case_2)

    #Get data subsets for cell specific flash populations
    wk_full,wk_fullq,wk_full_mask  = cell_subset(case_1,wk,wkq,'full')
    sl_full,sl_fullq,sl_full_mask  = cell_subset(case_2,sl,slq,'full')

    wk_left,wk_leftq,wk_left_mask  = cell_subset(case_1,wk,wkq,'left')
    sl_left,sl_leftq,sl_left_mask  = cell_subset(case_2,sl,slq,'left')

    wk_right,wk_rightq,wk_right_mask = cell_subset(case_1,wk,wkq,'right')
    sl_right,sl_rightq,sl_right_mask = cell_subset(case_2,sl,slq,'right')

    #Get integral and outer-length range lengths [Not used for capacitor paper]
    ll_wk_full,lr_wk_full,lo_wk_full,ln_wk_full = get_lengths(wk_full)
    ll_sl_full,lr_sl_full,lo_sl_full,ln_sl_full = get_lengths(sl_full)

    ll_wk_left,lr_wk_left,lo_wk_left,ln_wk_left = get_lengths(wk_left)
    ll_sl_left,lr_sl_left,lo_sl_left,ln_sl_left = get_lengths(sl_left)

    ll_wk_right,lr_wk_right,lo_wk_right,ln_wk_right = get_lengths(wk_right)
    ll_sl_right,lr_sl_right,lo_sl_right,ln_sl_right = get_lengths(sl_right)

    #Get dictionary objects for plotting histograms
    wk_dat_full  = get_vars(wk_full ,lo_wk_full ,ln_wk_full)
    wk_dat_right = get_vars(wk_right,lo_wk_right,ln_wk_right)
    wk_dat_left  = get_vars(wk_left ,lo_wk_left ,ln_wk_left)

    sl_dat_full  = get_vars(sl_full ,lo_sl_full ,ln_sl_full)
    sl_dat_right = get_vars(sl_right,lo_sl_right,ln_sl_right)
    sl_dat_left  = get_vars(sl_left ,lo_sl_left ,ln_sl_left)
    #-------------------------------------------------------------------------------------------------

    #get data subsets, and return individual dataframes
    (wk_df, wk_energy,wk_time,wk_init_alt,
            wk_separation,wk_ebrk,wk_area,wk_cap,wk_bins) = wk_data

    (sl_df, sl_energy,sl_time,sl_init_alt,
            sl_separation,sl_ebrk,sl_area,sl_cap,sl_bins) = sl_data


    #Get neutralization efficiences for COMMAS and capacitor flash energies, eta_m (Eq. 4) and eta_c (Eq. 2), respectively
    wk_commas_eta  = wk_df.eta_m
    sl_commas_eta  = sl_df.eta_m

    #Compute capacitor effiencies from energy -- eta_c in data file report previous
    #eta computed using a different (old) capacitor model configuration.
    wk_cap_eta     = wk_energy/wk_cap #wk_df.eta_c
    sl_cap_eta     = sl_energy/sl_cap #sl_df.eta_c

    #Get median values for use in adjustment of energy estimates for the capacitor model
    wk_commas_etaM = np.nanmedian(wk_commas_eta)
    sl_commas_etaM = np.nanmedian(sl_commas_eta)
    print('Median COMMAS neutralization efficiencies: {0} and {1} for WK82 and SL16,respectively'.
          format(wk_commas_etaM,sl_commas_etaM))

    if uniform_eta == False:
        wk_cap_etaM    = np.nanmedian(wk_cap_eta)
        sl_cap_etaM    = np.nanmedian(sl_cap_eta)
        print('Median Capacitor neutralization efficiencies: {0} and {1} for WK82 and SL16,respectively'.
              format(wk_cap_etaM,sl_cap_etaM))
    else:
        '''Select a value for eta_c to adjust energy'''
        wk_cap_etaM    = eta_u
        sl_cap_etaM    = eta_u
        print('Assuming A Uniform Capacitor neutralization efficiency: {0} and {1} for WK82 and SL16,respectively'.
              format(wk_cap_etaM,sl_cap_etaM))


    ######################
    #PRE ADJUSTMENT      #
    #######################################################################
    #Get minute bins of energy totals and averages
    (wk_tedges,wk_commas_totals,
     wk_commas_means,wk_cap_totals,wk_cap_means) = energy_time_series(wk_time,wk_df,wk_cap,wk_bins,wk_cap_etaM,False)

    (sl_tedges,sl_commas_totals,
     sl_commas_means,sl_cap_totals,sl_cap_means) = energy_time_series(sl_time,sl_df,sl_cap,sl_bins,sl_cap_etaM,False)


    wk = (wk_tedges,wk_commas_totals,wk_commas_means,wk_cap_totals,wk_cap_means)
    sl = (sl_tedges,sl_commas_totals,sl_commas_means,sl_cap_totals,sl_cap_means)
    t_centers,commas_totals,commas_means,cap_totals,cap_means = data_tuples(wk,sl)

    #PLOT ORIGINAL ETIMATES
    energy_compare(t_centers,commas_totals,commas_means,cap_totals,cap_means,(wk_cap_etaM,sl_cap_etaM),False)
    print('Plotted Capacitor and COMMAS flash energy change pre neutralization efficiency adjustment')

    ######################
    #ADJUSTMENT          #
    ########################################################################
    #Now plot adjusted energies:
    #Get minute bins of energy totals and averages
    (wk_tedges,wk_commas_totals,
     wk_commas_means,wk_cap_totals,wk_cap_means) = energy_time_series(wk_time,wk_df,wk_cap,wk_bins,wk_cap_etaM,True)

    (sl_tedges,sl_commas_totals,
     sl_commas_means,sl_cap_totals,sl_cap_means) = energy_time_series(sl_time,sl_df,sl_cap,sl_bins,sl_cap_etaM,True)


    wk = (wk_tedges,wk_commas_totals,wk_commas_means,wk_cap_totals,wk_cap_means)
    sl = (sl_tedges,sl_commas_totals,sl_commas_means,sl_cap_totals,sl_cap_means)
    t_centers,commas_totals,commas_means,cap_totals,cap_means = data_tuples(wk,sl)

    #PLOT ORIGINAL ETIMATES
    energy_compare(t_centers,commas_totals,commas_means,cap_totals,cap_means,(wk_cap_etaM,sl_cap_etaM),True)
    print('Plotted Capacitor and COMMAS flash energy change with neutralization efficiency adjustment')


    #######################
    #FLASH LENGTH ANALYSIS#
    ##########################################################################
    #Bin W and eta for COMMAS and Capacitor:
    #---------------------------------
    sl_len_w,percents_sl_w       = box_bins(sl_df.area,sl_cap)
    wk_len_w,percents_wk_w       = box_bins(wk_df.area,wk_cap)

    sl_len_wcom,percents_sl_wcom = box_bins(sl_df.area,sl_energy)
    wk_len_wcom,percents_wk_wcom = box_bins(wk_df.area,wk_energy)

    sl_len_eta,percents_sl_eta   = box_bins(sl_df.area,sl_cap_eta)
    wk_len_eta,percents_wk_eta   = box_bins(wk_df.area,wk_cap_eta)

    sl_len_etacom,percents_sl_etacom = box_bins(sl_df.area,sl_commas_eta)
    wk_len_etacom,percents_wk_etacom = box_bins(wk_df.area,wk_commas_eta)


    #tuple of binned data:
    wk_len_bins = (wk_len_w,wk_len_wcom,wk_len_eta,wk_len_etacom,percents_wk_w)
    sl_len_bins = (sl_len_w,sl_len_wcom,sl_len_eta,sl_len_etacom,percents_sl_w)

    box_plots(sl_len_bins,wk_len_bins)

    #######################
    #ETA TIME-SERIES      #
    #######################
    ###########################################################################
    eta_time_series(wk_df,sl_df,wk_cap_etaM,sl_cap_etaM,wk_cap,sl_cap,wk_bins,sl_bins)

    #######################
    #CASE PLOTS           #
    ###########################################################################
    print(wk_df.columns)
    wk_sim_time = wk_df.w_time
    sl_sim_time = sl_df.w_time

    case_time_series(wk_df,case_wk,wk_bins,wk_sim_time)
    case_time_series(sl_df,case_sl,sl_bins,sl_sim_time)

    case_energy_area(wk_df.area,wk_energy,'WK82')
    case_energy_area(sl_df.area,sl_energy,'SL16')

    ######################
    #Energy Covariance   #
    ######################
    cov_time_series(wk_cap,wk_energy,wk_time,wk_bins,wk_cap_etaM,sl_cap,sl_energy,sl_time,sl_bins,sl_cap_etaM)

    ######################
    #Energy Histograms   #
    ######################
    energy_hists(wk_dat_full,sl_dat_right)
