import numpy as np
import pandas as pd
import scipy.stats as st
from EnergyPlotting import hist_min
import itertools

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.reset_orig()

def make_patch_spines_invisible(ax):
    '''
    Spines on axes will be set to invisible on
    axes offset from parent figure.

    For use with figures with more than 2 shared
    axes.

    Adjusted from:
    https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    '''
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def case_time_series(df,case,bins,sim_time):
    t_edges, hist_mean_area = hist_min(df['Time (s)'],df.area                 ,bins,'means' )
    t_edges, hist_total_area= hist_min(df['Time (s)'],df.area                 ,bins,'totals')
    t_edges, hist_total_w   = hist_min(df['Time (s)'],-df[' Change in Energy'],bins,'totals')
    t_edges, hist_mean_w    = hist_min(df['Time (s)'],-df[' Change in Energy'],bins,'means' )
    t_edges, hist_total_rate= hist_min(df['Time (s)'],None                    ,bins,'totals') #Flash rate



    fig, ax = plt.subplots(2,2,figsize=(15,10))
    ax_rate1 = ax[0,0].twinx()
    ax_rate2 = ax[0,1].twinx()

    ax_eng1  = ax[1,0].twinx()
    ax_eng2  = ax[1,1].twinx()

    axwind   = ax[0,0].twinx()
    axwindb  = ax[0,1].twinx()

    axwindc  = ax[1,0].twinx()
    axwindd  = ax[1,1].twinx()


    #Fix Splines:
    #----------------------------------------------------
    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    axwind.spines["right"].set_position(("axes", 1.2))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(axwind)
    # Second, show the right spine.
    axwind.spines["right"].set_visible(True)

    axwindb.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(axwindb)
    axwindb.spines["right"].set_visible(True)

    axwindc.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(axwindc)
    axwindc.spines["right"].set_visible(True)

    axwindd.spines["right"].set_position(("axes", 1.2))
    make_patch_spines_invisible(axwindd)
    axwindd.spines["right"].set_visible(True)

    sort_t = np.argsort(np.unique(sim_time))

    #Line plots:
    #-----------------------------------------------------
    p1,= ax[0,0].plot(t_edges,hist_total_area,'C1' ,linewidth=4)
    p2,=ax_rate1.plot(t_edges,hist_total_rate      ,linewidth=3)
    p3,=axwind.plot((sim_time),(df.w_vol),'k--',linewidth=2 ,label='Updraft Volume')


    ax[0,1].plot( t_edges,hist_mean_area*1e0,'C1',linewidth=4)
    ax_rate2.plot(t_edges,hist_total_rate   ,'C0',linewidth=3)
    axwindb.plot((sim_time),(df.w_vol),'k--',linewidth=2  ,label='Updraft Volume')


    p1b, = ax[1,0].plot(t_edges,hist_total_area  ,'C1' ,linewidth=4)
    p2b, = ax_eng1.plot(t_edges,hist_total_w*1e-9,'C2' ,linewidth=3)
    p3b, = axwindc.plot((sim_time),(df.w_vol),'k--',linewidth=2  ,label='Updraft Volume')


    ax[1,1].plot(t_edges,hist_mean_area  ,'C1' ,linewidth=4)
    ax_eng2.plot(t_edges,hist_mean_w*1e-9,'C2' ,linewidth=3)
    axwindd.plot((sim_time),(df.w_vol),'k--',linewidth=2,  label='Updraft Volume')


    ax[0,0].legend([p1 ,p2 ,p3] ,['Total Flash Area','Flash Rate'        ,'Updraft Volume'],fontsize=12)
    ax[0,1].legend([p1 ,p2 ,p3] ,['Mean Flash Area' ,'Flash Rate'        ,'Updraft Volume'],fontsize=12)
    ax[1,0].legend([p1b,p2b,p3b],['Total Flash Area','Total Flash Energy','Updraft Volume'],fontsize=12)
    ax[1,1].legend([p1b,p2b,p3b],['Mean Flash Area' ,'Mean Flash Energy' ,'Updraft Volume'],fontsize=12)


    #Fix colors of axes
    ax[0,0].yaxis.label.set_color( p1.get_color())
    ax_rate1.yaxis.label.set_color(p2.get_color())
    axwind.yaxis.label.set_color(  p3.get_color())

    ax[0,1].yaxis.label.set_color( p1.get_color())
    ax_rate2.yaxis.label.set_color(p2.get_color())
    axwindb.yaxis.label.set_color( p3.get_color())

    ax[1,0].yaxis.label.set_color(p1b.get_color())
    ax_eng1.yaxis.label.set_color(p2b.get_color())
    axwindc.yaxis.label.set_color(p3b.get_color())

    ax[1,1].yaxis.label.set_color(p1b.get_color())
    ax_eng2.yaxis.label.set_color(p2b.get_color())
    axwindd.yaxis.label.set_color(p3b.get_color())


    #Set Labels:
    #----------------------------------------------------
    ax[0,0].set_ylabel(r'Total Flash Area $(km^{2})$',fontsize=15)
    ax[1,0].set_ylabel(r'Total Flash Area $(km^{2})$',fontsize=15)
    ax[0,1].set_ylabel(r'Mean Flash Area $(km^{2})$' ,fontsize=15)
    ax[1,1].set_ylabel(r'Mean Flash Area $(km^{2})$' ,fontsize=15)

    ax[0,0].set_xlabel(r'Model Time (s)',fontsize=15)
    ax[1,0].set_xlabel(r'Model Time (s)',fontsize=15)
    ax[0,1].set_xlabel(r'Model Time (s)',fontsize=15)
    ax[1,1].set_xlabel(r'Model Time (s)',fontsize=15)



    ax_rate1.set_ylabel(r'Flash Rate $(min^{-1})$'  ,fontsize=15)
    ax_rate2.set_ylabel(r'Flash Rate $(min^{-1})$'  ,fontsize=15)
    ax_eng1.set_ylabel( r'Total Flash Energy $(GJ)$',fontsize=15)
    ax_eng2.set_ylabel( r'Mean Flash Energy $(GJ)$' ,fontsize=15)


    axwind.set_ylabel( r'Updraft Volume ($km^{3}$)',fontsize=15)
    axwindb.set_ylabel(r'Updraft Volume ($km^{3}$)',fontsize=15)
    axwindc.set_ylabel(r'Updraft Volume ($km^{3}$)',fontsize=15)
    axwindd.set_ylabel(r'Updraft Volume ($km^{3}$)',fontsize=15)

    for (i,j) in itertools.product(range(2),range(2)):
        ax[i,j].grid(alpha=0.4)
        ax[i,j].tick_params(labelsize=14)


    #Fix ticks:
    #---------------------------------------------------------
    ax_rate1.tick_params(labelsize=14,colors=p2.get_color())
    ax_rate2.tick_params(labelsize=14,colors=p2.get_color())
    ax_eng1.tick_params( labelsize=14,colors=p2b.get_color())
    ax_eng2.tick_params( labelsize=14,colors=p2b.get_color())

    ax[0,0].tick_params(which='y',colors=p1.get_color())
    ax[0,1].tick_params(which='y',colors=p1.get_color())
    ax[1,0].tick_params(which='y',colors=p1b.get_color())
    ax[1,1].tick_params(which='y',colors=p1b.get_color())


    [ax[i,j].set_xlim(t_edges.min(),t_edges.max()) for i in range(2) for j in range(2)]




    if case == 'WK':
        axwind.set_ylim( .5e3,2550)
        axwindb.set_ylim(.5e3,2550)
        axwindc.set_ylim(.5e3,2550)
        axwindd.set_ylim(.5e3,2550)

        for i,(a,l) in enumerate(zip(ax.flatten(),['A)','B)','C)','D)'])):
            if i == 0:
                a.annotate(l,xy=(1500,81000),fontsize=26,color='red',weight='bold')
            elif i == 1:
                a.annotate(l,xy=(1500,350)  ,fontsize=26,color='red',weight='bold')
            elif i == 2:
                a.annotate(l,xy=(1500,81000),fontsize=26,color='red',weight='bold')
            elif i == 3:
                a.annotate(l,xy=(1500,350)  ,fontsize=26,color='red',weight='bold')

    else:
        for i,(a,l) in enumerate(zip(ax.flatten(),['A)','B)','C)','D)'])):
            if i == 0:
                a.annotate(l,xy=(2000,44000),fontsize=26,color='red',weight='bold')
            elif i == 1:
                a.annotate(l,xy=(8200,90),fontsize=26,color='red',weight='bold')
            elif i == 2:
                a.annotate(l,xy=(2000,44000),fontsize=26,color='red',weight='bold')
            elif i == 3:
                a.annotate(l,xy=(8200,90),fontsize=26,color='red',weight='bold')

    plt.tight_layout()
    plt.savefig(f'Figures/{case}_SUMMARY.pdf',dpi=120,bbox_inches='tight')
    plt.close()

def case_energy_area(df_area,df_energy,case):
    import scipy.stats as stats

    fig,ax=plt.subplots(1,1,figsize=(6,6))
    rstat = stats.pearsonr(np.log10(df_area),np.log10(df_energy))
    g = sns.regplot(np.log10(df_area),np.log10(df_energy),color='k',scatter=False,ax=ax)
    sns.scatterplot(np.log10(df_area),np.log10(df_energy),color='C3',edgecolor='k',linewidth=.4,alpha=0.7)
    g.annotate(r'pearsonr{0:.2f}; p={1:.2f}'.format(rstat[0],rstat[1]),(-.25,9.5),fontsize=13)
    ax.tick_params(labelsize=14)
    ax.set_xlabel(r'$\rm log_{10}$($\rm \sqrt{A}$ [km])',fontsize=15)
    ax.set_ylabel(r'$\rm log_{10}$($\rm \Delta W_m \ [J]$)',fontsize=15)
    plt.savefig(f'Figures/{case}_AREA-ENERGY.pdf',dpi=120,bbox_inches='tight')
    plt.close()
