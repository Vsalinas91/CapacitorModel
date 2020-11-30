##################################################################
# Plotting Routines for Capacitor vs COMMAS Energy Comparisons:  #
# -------------------------------------------------------------  #
#                                                                #
# All Figures found in the manuscript may be reproduced using    #
# the routines below. The following will generate figures for    #
# the figure numbers found in text:                              #
#                                                                #
#     -) energy_compare(adjust=False) - Figure 3.                #
#     -) energy_compare(adjust=True)  - Figure 4.                #
#     -) box_plots                    - Figure 5.                #
#     -) eta_time_series              - Figure 6.                #
#                                                                #
# NOTE: Additional figures may be generated if the user decides  #
# to use a uniform eta_u value; defined in the CapacitorModelAna-#
# lysis script. These figures will reflect adjustment to capacit-#
# or energy values using a user specified neutralization efficie-#
# ncy.                                                           #
##################################################################

import numpy as np
import pandas as pd
import scipy.stats as st

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_orig()


#################################################
#Specify Case Provided in Datasets: (WK82, SL16)#
#################################################
bin_range = 60. #for one minute bins - can change for other desired bin spacings

def hist_min(t,w,bins,which):
    '''Minute Total and Average Histograms

          t = initiation time
          w = weighted field for minute averages/totals
          bins  = number of bins to generate 60 second intervals
          which = totals or means are acceptable arguements

    '''
    hist   = np.histogram(t,bins=bins)
    histws = np.histogram(t,bins=bins,weights=w)

    if which == 'totals':
        t_series = histws[0]
    elif which == 'means':
        t_series = (histws[0])/(hist[0])
    else:
        assert (which == 'totals') or (which=='means'), 'which should be either "totals" or "means" only'
    return((histws[1][1:]+histws[1][:-1])/2.,t_series)


def box_bins(areas,values):
    '''
    Flash length binning: Bin any data field within specific flash length bins.

        -) Flash length = sqrt(flash_area)
        -) Bins log scaled:
              -1   = 0.10  km
              -0.5 = 0.316 km
              0.0  = 1.0   km
              0.5  = 3.16  km
              1.0  = 10    km
              1.5  = 31.6  km

    Returns dataframe with data grouped within length bin columns, and % values of flashes within
    each bin (e.g. fracN where N is the bin number from 0 to 5).
    '''
    l    = np.log10((areas)**0.5) #log scaled
    b0   = (l <= -1)
    b1   = (l > -1)   & (l <= -0.5)
    b2   = (l > -0.5) & (l <= 0)
    b3   = (l > 0)    & (l <= 0.5)
    b4   = (l > 0.5)  & (l <= 1)
    b5   = (l >1)     & (l<=1.5)

    v0   = (values[b0])
    v1   = (values[b1])
    v2   = (values[b2])
    v3   = (values[b3])
    v4   = (values[b4])
    v5   = (values[b5])

    df   = pd.DataFrame([v0,v1,v2,v3,v4,v5]).T
    df.columns = [r'$\leq$ 0.10', '0.316', '1.0', '3.16', '10.0',r'$\geq$ 31.6']


    frac0  = v0.shape[0]/values.shape[0]
    frac1  = v1.shape[0]/values.shape[0]
    frac2  = v2.shape[0]/values.shape[0]
    frac3  = v3.shape[0]/values.shape[0]
    frac4  = v4.shape[0]/values.shape[0]
    frac5  = v5.shape[0]/values.shape[0]
    return(df,[frac0,frac1,frac2,frac3,frac4,frac5])

def temporal_binning(df_cap,df_comm,df_time,df_time_bins,df_etac):
    df_minutes = []
    df_com_minutes = []
    for i in range(len(df_time_bins)):
        if  df_time_bins[i] == df_time_bins[0]:
            df_minutes.append(df_cap[df_time<=df_time_bins[0]]*(df_etac) * 1e-9)
            df_com_minutes.append(df_comm[df_time<=df_time_bins[0]] * 1e-9)
        elif df_time_bins[i] == df_time_bins[-1]:
            df_minutes.append(df_cap[df_time>=df_time_bins[-1]]*(df_etac) * 1e-9)
            df_com_minutes.append(df_comm[df_time>=df_time_bins[-1]] * 1e-9)
        else:
            df_minutes.append(df_cap[(df_time_bins[i] < df_time) & (df_time <= df_time_bins[i+1])]*(df_etac) * 1e-9)
            df_com_minutes.append(df_comm[(df_time_bins[i] < df_time) & (df_time <= df_time_bins[i+1])] * 1e-9)

    return(np.array(df_minutes),np.array(df_com_minutes))

def temporal_covariance(df_cap,df_comm,df_time,df_time_bins,df_etac):
    covariance = temporal_binning(df_cap,df_comm,df_time,df_time_bins,df_etac)
    #Normalized if bias = 1, else bias = 0 by default
    cov_xy     = [np.cov(covariance[0][i],covariance[1][i])[0][1] for i in range(covariance[0].shape[0])] #bias = 1 for * (covariance[0].shape[0]-1)
    return(cov_xy)


def energy_time_series(init_time,df,cap,minute_bins,eta,adjust):
    '''
    Get one minute (or specified time intervals) binned data fields. Used for
    flash energy comparisons, and summary figures.

    Returns:
        -) time bin centers from edges - t_edges
        -) commas flash energy change minute totals - commas_totals
        -) commas flash energy change minute averages-commas_means
        -) capacitor flash energy minute totals - cap_totals
        -) capacitor flash energy minute averages-cap_means
    '''
    #COMMAS TOTALS:
    t_edges, commas_totals = hist_min( init_time, -df[' Change in Energy'], minute_bins, 'totals')
    #COMMAS AVERAGES:
    t_edges, commas_means  = hist_min( init_time, -df[' Change in Energy'], minute_bins, 'means')
    if adjust == False:
        #Totals:
        t_edges, cap_totals= hist_min( init_time, cap, minute_bins, 'totals')
        #Averages:
        t_edges, cap_means = hist_min( init_time, cap, minute_bins, 'means')
    else:
        #Totals
        t_edges, cap_totals= hist_min( init_time, cap*eta, minute_bins, 'totals')
        #Averages:
        t_edges, cap_means = hist_min( init_time, cap*eta, minute_bins, 'means')
    return(t_edges,commas_totals,commas_means,cap_totals,cap_means)


def energy_compare(t_centers,commas_totals,commas_means,cap_totals,cap_means,eta,adjust):
    '''
    Plot the energy comparisons between COMMAS and Capacitor flash energy neutralizations.
    Arguments should be tuples in form:

        t_centers    = (wk_t_centers    ,sl_t_centers    )
        commas_totals= (wk_commas_totals,sl_commas_totals)
        commas_means = (wk_commas_means ,sl_commas_means )
        cap_totals   = (wk_cap_totals   ,sl_cap_totals   )
        cap_means    = (wk_cap_means    ,sl_cap_means    )

        adjusted     = if neutralization efficiency is applied TRUE else FALSE.
                       New axes drawn if FALSE to compare energy values > 1 order of magnitude
                       larger than COMMAS values.

        Figures saved to Figures directory upon running method.

    '''
    #Get data for cases:
    wk_t_centers,sl_t_centers = t_centers

    wk_commas_totals,sl_commas_totals = commas_totals
    wk_cap_totals,sl_cap_totals       = cap_totals

    wk_commas_means,sl_commas_means   = commas_means
    wk_cap_means,sl_cap_means         = cap_means

    fig,ax = plt.subplots(2,2,figsize=(15,10))
    if adjust == False:
        ax2 = ax[0,0].twinx()
        ax3 = ax[0,1].twinx()
        ax4 = ax[1,0].twinx()
        ax5 = ax[1,1].twinx()
    else:
        ax2 = ax[0,0]
        ax3 = ax[0,1]
        ax4 = ax[1,0]
        ax5 = ax[1,1]


    # #Average Time Series: WK TOTALS
    # #------------------------------------------------------------------------------------------
    wkC_totals, = ax[0,0].plot(wk_t_centers,wk_cap_totals*1e-9   ,'C3'  ,label='Capacitor Energy')
    wkD_totals, = ax2.plot(    wk_t_centers,wk_commas_totals*1e-9,'C0--',label='COMMAS Energy')
    #End --------------------------------------------------------------------------------------

    #Average Time Series: WK MEANS
    #------------------------------------------------------------------------------------------
    wkC_means,  = ax[1,0].plot(wk_t_centers,wk_cap_means*1e-9    ,'C3'  ,label='Capacitor Energy')
    wkD_means,  = ax4.plot(    wk_t_centers,wk_commas_means*1e-9,'C0--',label='COMMAS Energy')
    # #End --------------------------------------------------------------------------------------

    #Average Time Series: KTAL Totals
    #------------------------------------------------------------------------------------------
    slC_totals, = ax[0,1].plot(sl_t_centers,sl_cap_totals*1e-9   ,'C3'  ,label='Capacitor Energy')
    slD_totals, = ax3.plot(    sl_t_centers,sl_commas_totals*1e-9,'C0--',label='COMMAS Energy')
    #End --------------------------------------------------------------------------------------

    #Average Time Series: KTAL MEANS
    #------------------------------------------------------------------------------------------
    slC_means,  = ax[1,1].plot(sl_t_centers,sl_cap_means*1e-9   ,'C3'  ,label='Capacitor Energy')
    slD_means,  = ax5.plot(    sl_t_centers,sl_commas_means*1e-9,'C0--',label='COMMAS Energy')
    #End --------------------------------------------------------------------------------------


    if adjust == False:
        ylabel = 'Electrostatic Energy Change (GJ)'
        for i, axs in enumerate((ax2,ax3,ax4,ax5)):
            axs.set_ylabel('COMMAS Energy (GJ)',fontsize=15)
            axs.tick_params(axis='both', which='major', labelsize=12)
    else:
        ylabel = 'Capacitor Energy (GJ)'
        ax[0,0].set_title(r'$\tilde{{\eta}}_c$ = {0:.3f}'.format(eta[0]),fontsize=17)
        ax[0,1].set_title(r'$\tilde{{\eta}}_c$ = {0:.3f}'.format(eta[1]),fontsize=17)


    #Axis Labels:
    [ax[i,j].set_xlabel('Model Time (s)'       ,fontsize=15) for i in range(2) for j in range(2)]
    [ax[i,j].set_ylabel(ylabel,fontsize=15) for i in range(2) for j in range(2)]
    #Tick Parameters:
    [ax[i,j].tick_params(axis='both', which='major', labelsize=12) for i in range(2) for j in range(2)]



    #TOP and BOTTOM LEGENDS:
    ax[0,0].legend([wkC_totals,wkD_totals],
                   ['WK82 Total Capacitor Energy','WK82 Total COMMAS Energy'],fontsize=14,loc='upper left')
    ax[0,1].legend([slC_totals,slD_totals],
                   ['SL16 Total Capacitor Energy','SL16 Total COMMAS Energy'],fontsize=14,loc='upper left')
    ax[1,0].legend([wkC_means,wkD_means],
                   ['WK82 Mean Capacitor Energy','WK82 Mean COMMAS Energy'],fontsize=14)
    ax[1,1].legend([slC_means,wkD_means],
                   ['SL16 Mean Capacitor Energy','SL16 Mean COMMAS Energy'],fontsize=14)

    #Annotate:
    #-----------------------
    for i,(a,l) in enumerate(zip([ax2,ax3,ax4,ax5],['A)','B)','C)','D)'])):
        if i==0:
            a.annotate(l,xy=(1500,450),fontsize=26,color='red',weight='bold')
        elif i==2:
            a.annotate(l,xy=(1500,2.4),fontsize=26,color='red',weight='bold')
        elif i==1:
            a.annotate(l,xy=(1800,170),fontsize=26,color='red',weight='bold')
        elif i==3:
            a.annotate(l,xy=(1800,3.5),fontsize=26,color='red',weight='bold')


    plt.tight_layout()
    if adjust == False:
        plt.savefig('Figures/ENERGY_COMPARE.pdf',bbox_inches='tight')
    else:
        plt.savefig('Figures/ENERGY_COMPARE_ETA.pdf',bbox_inches='tight')

def violin_alpha(ax,alpha):
    '''
    Set alpha for violin plots using the axis collection of violin objects.
    Seaborn currently doesn't have a keyword arguement to handle this.
    '''
    for violin in ax.collections:
        violin.set_alpha(alpha)

def box_plots(sl_bins,wk_bins): #FIX
    '''
    Reproduce Violin Plots for eta and energy distributions as a function of length.
    Flash lengths bins are defined in the box_bins() method.
    '''

    #get energy and percent of flashes for each length bin:
    sl_len_cap,sl_len_com,sl_cap_eta,sl_com_eta,percents_sl = sl_bins
    wk_len_cap,wk_len_com,wk_cap_eta,wk_com_eta,percents_wk = wk_bins

    median_eta_wk = np.nanmedian(wk_cap_eta)
    median_eta_sl = np.nanmedian(sl_cap_eta)
    #Axis bins, these are relative which index the violine is being plotted at.
    bin_range = [0,1,2,3,4,5]

    fig,ax = plt.subplots(2,2,figsize=(12,11))

    plot_type = sns.boxplot#sns.violineplot
    fliers    = True

    #ETA:
    #-------------------------------------------------------------------
    #COMMAS
    v1com = plot_type(data=np.log10(sl_com_eta),ax=ax[0,1],color='tab:blue',width=.5,boxprops=dict(alpha=.6),showfliers = fliers);
    v2com = plot_type(data=np.log10(wk_com_eta),ax=ax[0,0],color='tab:blue',width=.5,boxprops=dict(alpha=.6),showfliers = fliers);

    #Empty plots; use for legends
    dcom, = ax[0,0].plot([0,0],[0,0],color='tab:red' ,linewidth=3)
    dcap, = ax[0,0].plot([0,0],[0,0],color='tab:blue',linewidth=3)
    ax[0,0].legend([dcom,dcap],['Capacitor','COMMAS'],loc='upper center',fontsize=14)

    #Capacitor:
    v1cap = plot_type(data=np.log10(sl_cap_eta),ax=ax[0,1],color='red',width=.5,boxprops=dict(alpha=.6),showfliers = fliers);
    v2cap = plot_type(data=np.log10(wk_cap_eta),ax=ax[0,0],color='red',width=.5,boxprops=dict(alpha=.6),showfliers = fliers);

    alpha = 0.6
    #Set violin alpha:
    violin_alpha(v1com,alpha)
    violin_alpha(v2com,alpha)
    violin_alpha(v1cap,alpha)
    violin_alpha(v2cap,alpha)

    #Annotate fraction of flashes per length bin
    for i,(n) in enumerate(percents_wk):
        ax[0,0].annotate('{0:.2f}%'.format(n*100),xy=(bin_range[i]-.35,-7.),weight='bold',color='k',fontsize=11.5);
        ax[0,1].annotate('{0:.2f}%'.format(percents_sl[i]*100),xy=(bin_range[i]-.35,-7.),weight='bold',color='k',fontsize=11.5);

    ax[0,0].plot([-1,6],[0,0],'k-')
    ax[0,1].plot([-1,6],[0,0],'k-')


    #Energy:
    #-------------------------------------------------------------------
    #COMMAS
    e1com = plot_type(data=np.log10(sl_len_com),ax=ax[1,1],color='tab:blue',width=.5,boxprops=dict(alpha=.6),showfliers = fliers);
    e2com = plot_type(data=np.log10(wk_len_com),ax=ax[1,0],color='tab:blue',width=.5,boxprops=dict(alpha=.6),showfliers = fliers);
    #Capacitor:
    e1cap = plot_type(data=np.log10(sl_len_cap),ax=ax[1,1],color='red',width=.5,boxprops=dict(alpha=.6),showfliers = fliers);
    e2cap = plot_type(data=np.log10(wk_len_cap),ax=ax[1,0],color='red',width=.5,boxprops=dict(alpha=.6),showfliers = fliers);


    #-5/3 line:
    ax[1,0].plot(np.linspace(1,5,6),np.linspace(6,11,6),'k:')
    ax[1,1].plot(np.linspace(1,5,6),np.linspace(6,11,6),'k:')


    #Set violin alpha for energy
    violin_alpha(e1com,alpha)
    violin_alpha(e2com,alpha)
    violin_alpha(e1cap,alpha)
    violin_alpha(e2cap,alpha)

    [ax[0,j].set_ylim(-8,8) for j in range(2)]
    [ax[1,j].set_ylim(0,13) for j in range(2)]
    ax[1,0].set_xlim(-1,6)
    ax[1,1].set_xlim(-1,6)

    [ax[0,j].set_ylabel(r'$\rm log_{10}(\eta)$',fontsize=15)          for j in range(2)]
    [ax[1,j].set_ylabel(r'$\rm log_{10}(W [J])$',fontsize=15)         for j in range(2)]
    [ax[i,j].set_xlabel(r'$\rm \sqrt{A} [km]$',fontsize=15)           for i in range(2) for j in range(2)]
    [ax[i,j].tick_params(labelsize=13) for i in range(2)              for j in range(2)]
    [ax[i,j].grid(alpha=0.3) for i in range(2)                        for j in range(2)]



    ax[0,0].set_title('WK82 Flashes',fontsize=15,weight='bold')
    ax[0,1].set_title('SL16 Flashes',fontsize=15,weight='bold')


    ax[0,0].plot([-1,6],[np.log10(median_eta_wk),np.log10(median_eta_wk)],'k--')
    ax[0,1].plot([-1,6],[np.log10(median_eta_sl),np.log10(median_eta_sl)],'k--')


    #Current annotation method is not well handled. Should use transAxes instead.
    for i,(a,l) in enumerate(zip(ax.flatten(),['A)','B)','C)','D)'])):
        if i<2:
            a.annotate(l,xy=(4.8,6),fontsize=26,color='red',weight='bold')
        elif i>=2 and i<4:
            a.annotate(l,xy=(-.3,11.5),fontsize=26,color='red',weight='bold')
        elif i >=4:
            a.annotate(l,xy=(-5.5,.9),fontsize=26,color='red',weight='bold')

    plt.tight_layout()
    plt.savefig('Figures/ETA_ENERGY_FLASH-LENGTHS.pdf',dpi=180,bbox_inches='tight')


def eta_time_series(wk,sl,wk_eta,sl_eta,wk_cap,sl_cap,wk_bins,sl_bins): #FIX
    '''
    Reproduce eta ration time series -- sum(w_cap)/sum(w_commas) as a function of time.
    For consistency, these values are binned every minute as with the energy comparisons,
    and summary figures.

    Arguements: wk,sl = wk and sl initiation dataframes.
    '''
    wk_df = wk
    sl_df = sl

    fig,ax = plt.subplots(1,1,figsize=(13,5))
    axb = ax.twiny()

    wk_etaM = np.nanmedian(wk_eta)
    sl_etaM = np.nanmedian(sl_eta)

    #Get Total Energy Fractions per minute.
    wk_etot_cap, wk_e_histtot_cap = hist_min( wk_df['Time (s)'],(wk_cap*wk_etaM)              ,wk_bins,'totals')
    wk_etot_com, wk_e_histtot_com = hist_min( wk_df['Time (s)'],(-wk_df[' Change in Energy']),wk_bins,'totals')

    sl_etot_cap, sl_e_histtot_cap = hist_min( sl_df['Time (s)'],(sl_cap*sl_etaM)              ,sl_bins,'totals')
    sl_etot_com, sl_e_histtot_com = hist_min( sl_df['Time (s)'],(-sl_df[' Change in Energy']),sl_bins,'totals')

    #Plot total energy fraction time series
    wkts, = ax.plot( wk_etot_cap, 10**np.log10(wk_e_histtot_cap/wk_e_histtot_com)*100,'-C0',linewidth=3)
    slts, = axb.plot(sl_etot_cap, 10**np.log10(sl_e_histtot_cap/sl_e_histtot_com)*100,'-C3',linewidth=3)

    ax.legend([wkts,slts],['WK82','SL16'],fontsize=14)

    #format time series
    axb.plot([sl_etot_cap.min(),sl_etot_cap.max()],[100,100],'k-')
    ax.set_xlim( wk_df['Time (s)'].min(),wk_df['Time (s)'].max())
    axb.set_xlim(sl_df['Time (s)'].min(),sl_df['Time (s)'].max())

    ax. set_xlabel('WK82 Simulation Time [s]',fontsize=15,color='C0')
    axb.set_xlabel('SL16 Simulation Time [s]',fontsize=15,color='C3')
    ax.set_ylabel(r'$\Sigma \eta_c W_c$ / $\Sigma W_m$',fontsize=15)
    ax. tick_params(labelsize=14)
    axb.tick_params(labelsize=14)
    #Annotate the total energy fraction(s) for entire durations
    wk_percent = (wk_cap*wk_etaM).sum()/-wk_df[' Change in Energy'].sum()
    sl_percent = (sl_cap*sl_etaM).sum()/-sl_df[' Change in Energy'].sum()

    ax.annotate(r'$\Sigma \eta_c W_c$ / $\Sigma W_m$ = {0:.2f} %'.format(wk_percent*100),xy=(3400,55), color='tab:blue',fontsize=16,weight='bold')
    ax.annotate(r'$\Sigma \eta_c W_c$ / $\Sigma W_m$ = {0:.2f} %'.format(sl_percent*100),xy=(3400,21), color='tab:red', fontsize=16,weight='bold')
    ax.grid(axis='y',alpha=0.3)
    plt.savefig('Figures/WC_WM_RATIO.pdf',dpi=150,bbox_inches='tight')


def cov_time_series(wk_cap,wk_comm,wk_time,wk_time_bins,wk_etac,sl_cap,sl_comm,sl_time,sl_time_bins,sl_etac):
    wk_cov = temporal_covariance(wk_cap,wk_comm,wk_time,wk_time_bins,wk_etac)
    sl_cov = temporal_covariance(sl_cap,sl_comm,sl_time,sl_time_bins,sl_etac)

    fig,ax = plt.subplots(1,1,figsize=(12,5))
    axb = ax.twiny()
    wk, = ax.plot(wk_time_bins,wk_cov,linewidth=3)
    sl, = axb.plot(sl_time_bins,sl_cov,'C3',linewidth=3)

    ax.legend([wk,sl],['WK82','SL16'],fontsize=14)



    ax.tick_params(labelsize=14)
    axb.tick_params(labelsize=14)
    ax.grid(axis='y',alpha=0.3)
    ax.set_ylabel(r'cov($W_{d}$,$\ \Delta W_m$)',fontsize=15)

    ax.plot([800,6500],[0,0],'k')
    ax.set_xlim(wk_time_bins.min(),wk_time_bins.max())
    ax.set_xlabel('WK82 Simulation Time [s]',fontsize=15,color='C0')
    axb.set_xlabel('SL16 Simulation Time [s]',fontsize=15,color='C3')
    plt.savefig('Figures/CAP_COMM_COV.pdf',dpi=150,bbox_inches='tight')
