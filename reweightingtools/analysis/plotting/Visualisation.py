#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:55:10 2023

@author: schaefej51
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as tick
# 


def plotTrajPropPot(trajectory, Potential, trajinpot, 
                    analytic_Boltzmann, n_bins, 
                    delta      = 1000                    , 
                    tconv      = 10**(-4)                , 
                    x_line     = np.linspace(-2, 2, 1000),
                    figuresize = (6,7)  
                           ): 
    
            end    = len(trajectory)*tconv
            x_t    = np.arange(0,end,tconv)

            hist, bins2 = np.histogram(trajectory, n_bins, density=True)
            bin_array   = bins2[:-1] + (bins2[-1] - bins2[-2])/2
            boltz       = analytic_Boltzmann(Potential, bin_array)
            ############################################################        
            
            ht   = plt.figure(constrained_layout=True, figsize=figuresize)
            spec = ht.add_gridspec(ncols=2, nrows=9, figure=ht)
            #-------------------------------------
            ax_his = ht.add_subplot(spec[0:4, 1])      
            ax_his.plot(bin_array, hist, c='C0',label='dtraj') 
            ax_his.plot(bin_array, boltz,'--', c='C1',label='analy')
            ax_his.grid(True, axis='y')
            ax_his.grid(True, axis='x',color='C7', lw=1)
            ax_his.tick_params(direction='in', labelbottom=False)
            ax_his.set_ylabel(r'$\mu(x)$', fontsize=15)
            ax_his.legend()
            ax_his.set_xlim((min(trajectory), max(trajectory)))
#ax_his.vlines(0.8,0, max(hist), color='C1', lw=2)
#ax_his.vlines(-0.8,0,max(hist), color='C1', lw=2)
            #-------------------------------------
            ax_ts = ht.add_subplot(spec[0:7, 0])      
            ax_ts.plot(trajectory[::5*delta], x_t[::5*delta], '.', color='C0', lw=0.4)
            ax_ts.grid(True, axis='x',color='C7', lw=1)
            ax_ts.set_ylabel(r'time lagged timestep', fontsize=15)
            ax_ts.set_xlim(ax_his.get_xlim())
#ax_ts.tick_params(direction='in', labelbottom=False)
            ax_ts.set_xlabel(r'$x$', fontsize=15)
            #-------------------------------------
            ax_pot = ht.add_subplot(spec[4:7, 1])      
            ax_pot.plot(x_line,Potential.potential(x_line), color='C1',label='analy')
            ax_pot.scatter(trajectory[::101*delta],trajinpot, color='C0',label='dtraj')
            ax_pot.legend()
            #-------------------------------------
            ax_pot.set_xlabel(r'$x$', fontsize=15)
            ax_pot.grid(True, axis='x',color='C7', lw=1)
            ax_pot.set_ylabel(r'$U(x)$', fontsize=15)
            ax_pot.set_xlim(ax_his.get_xlim())
            ax_pot.set_ylim((min(trajinpot)-0.5, max(trajinpot)+0.5))
#ax_pot.vlines(0.8,0, max(trajinpot), color='C1', lw=2)
#ax_pot.vlines(-0.8,0,max(trajinpot), color='C1', lw=2)



#plt.savefig('/home/schaefej51/Documents/...'.png' , bbox_inches='tight', format='png')
            #plt.show()
            ht.tight_layout()
        
        
def plotMSMeigenvectors(eigenvectors, hist, bin_array, ana_boltzmann, n_bins):    
    fig, axes = plt.subplots(1, 4, figsize=(26, 5))
    for ax, i in zip(axes.flat, range(eigenvectors.shape[1])): 
        if i == 0:
            ax.plot(bin_array, ana_boltzmann, c='C1')
            ax.plot(bin_array, hist,'--', c='C2')
            ax.set_ylabel(r'propability density', fontsize=23)
        eigenvec=-eigenvectors[i,:]/max(abs(eigenvectors[0,:]))
        ax.plot(bin_array,eigenvec) #,label='$eigvec$'+str(i))

        ax.set_xlabel(r'position $x$', fontsize=23)
    
        ax.tick_params(axis='both', which='major', labelsize=18)
    
        d = np.zeros(len(bin_array))
        ax.fill_between(bin_array, eigenvec, where=-eigenvectors[i,:]>=d, interpolate=True, color='C0', alpha=0.3)
        ax.fill_between(bin_array, eigenvec, where=-eigenvectors[i,:]<=d, interpolate=True, color='C4', alpha=0.3)

def plotMSMeigenvalues(Lambda,tau):
    if tau.size==1:
        ev     = plt.figure(figsize=(3.75*tau.size,3.75*tau.size))
    else:
        ev     = plt.figure(figsize=(3.75*tau.size,tau.size))
    spec5  = gridspec.GridSpec( 1,tau.size,  hspace=0.4, wspace=0.2)
    a      = list()
    images = list()

    for col in range(tau.size):
        ax = ev.add_subplot(spec5[0, col])
        if col==0:
            ax.set_ylabel(r'eigenvalue',fontsize=15)
        ax.set_title(r'$\tau$= '+str(tau[col]), fontsize=15)
        ax.set_xlabel(r'index of eigenvalue',fontsize=15)
        

        x = np.linspace(0,Lambda[col].size,Lambda[col].size)
        y = Lambda[col]

        plt.stem(x, y, use_line_collection=True)
        #plt.savefig('/home/schaefej51/Documents/Peptide/RESULTS/msm_1D_2_lamb.png' , bbox_inches='tight', format='png')
        
def plotITS(ITS, tau):
    its     = plt.figure(figsize=(12,7))
    spec5  = gridspec.GridSpec(1,1, hspace=0.4, wspace=0.4)
    Pic=[]
    ax = its.add_subplot(spec5[0, 0])
    ax.set_xlabel(r'lag-time $\tau$', fontsize=23)
    ax.set_ylabel(r'implied time scale $t_i$', fontsize=23)
    for p in range(3):
        pic = ax.plot(np.array(tau),ITS[:,p],lw=3,label=str(p))
        Pic.append(pic[0])
    ax.plot(tau,np.array(tau)*2, color='C7', lw=0.4)
    H = np.array(tau)*2
    d = np.zeros(len(tau))
    ax.fill_between(tau, H, where=H>=d, interpolate=True, color='C7')
    ax.legend(Pic,['2. left eigenvalue','3. left eigenvalue','4. left eigenvalue','5. left eigenvalue','6. left eigenvalue','7. left eigenvalue'],loc='center right', fontsize=23)

    ax.tick_params(axis='both', which='major', labelsize=18)

    #plt.savefig('/home/schaefej51/Documents/Peptide/RESULTS/msm_1D_2_its.png' , bbox_inches='tight', format='png')
    

def plot_implied_timescales_grids(grids,
                                  n_trajs,
                                  directory,
                                  Dim,
                                  shared_discretisation,
                                  time_step,
                                  lagtimes,
                                  k,
                                  save=False
                                 ):
        
    its_grids = []
    for g in grids: 
        its_trajs = []
        for t in range(n_trajs):
            if Dim=='1DX':
                file_name = '1DX_'+str(t) + '_' + str(g)
            elif Dim=='1DY':
                file_name = '1DY_'+str(t) + '_' + str(g)
            elif Dim=='2D':
                file_name = '2D_'+str(t) + '_' + str(g)
            
            if shared_discretisation:
                its_trajs.append(np.loadtxt(directory +'mp_'+ file_name + ".dat"))
            else:
                its_trajs.append(np.loadtxt(directory + file_name + ".dat"))
        its_grids.append(its_trajs)
    its_data = np.array(its_grids) 
    
    its_mean = np.mean(its_data, axis=1) *time_step
    its_std  = np.std(its_data, axis=1) *time_step
    lagtimes = np.array(lagtimes) *time_step
    
    fig, a = plt.subplots(3,2, figsize=(15,15), sharex=True, sharey=True)
    #fig.suptitle('Implied timescales',x=0.53,y=0.9,fontsize=17)
    a = a.ravel()
    for idx, ax in enumerate(a):
        if idx!=1:
          ix = idx
          if idx > 1:
              ix = idx -1
        mean = its_mean[ix]
        std  = its_std[ix] 

        grid_size = grids[ix]*time_step
        ax.axvline(grid_size, ymin=0.05, 
                     ymax=mean[:,0].max()+std[:,0].max(),color='C7', lw=0.4)
        ax.fill_betweenx(np.linspace(0.0,mean[:,0].max()+0.25, len(lagtimes)), 0, 
                          grid_size, interpolate=True, 
                          color='C7', alpha=0.1)  

        ax.set_title('grid size: ' +str(grids[ix]), fontsize=17)
        for ev in range(k-1):
            if idx!=1:
                ax.axhline(mean[:,ev].mean(), xmin=0.0, 
                           xmax=0.95,color='C7', lw=0.4)
                
       
            ax.plot(lagtimes, mean[:,ev] ,'--',  c='C'+str(ev), 
                    lw=3,label=str(ev+1)+'. slowest process')
            ax.fill_between(lagtimes,mean[:,ev]-std[:,ev],
                            mean[:,ev]+std[:,ev], 
                            color='C'+str(ev),alpha=.1)
               
        ax.plot(lagtimes,np.array(lagtimes)*2, color='C7', lw=0.4)
        H = np.array(lagtimes)*2
        d = np.zeros(len(lagtimes))
        ax.fill_between(lagtimes, H, where=H>=d, interpolate=True, 
                            color='C7', alpha=0.1)
        
        
        
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        if idx % 2 == 0:
            ax.set_ylabel('timescale in ps', fontsize=20, labelpad=15)
        if idx == 5 or idx == 5-1:
            ax.set_xlabel('lag time in ps', fontsize=20, labelpad=15)
        if idx == 0:
            ax.legend(title='Implied timescales', title_fontsize=20,
                      loc='center left', bbox_to_anchor=(1.25, 0.5), fontsize=20)
     # plt.tight_layout()#pad=1, w_pad=0., h_pad=0.)                                     
    fig.delaxes(a[1]) 
    if save:
        if Dim=='1DX':
            fig.savefig(directory + 'its_grid_sizes_1DX.png' , bbox_inches='tight', format='png')      
        elif Dim=='1DY':
            fig.savefig(directory + 'its_grid_sizes_1DY.png' , bbox_inches='tight', format='png')      
        elif Dim=='2D':
            fig.savefig(directory + 'its_grid_sizes.png' , bbox_inches='tight', format='png')      
        
def plot_implied_timescales(n_trajs,
                            store_directory,
                            implied_timescales,
                            Dim,
                            shared_discretisation,
                            time_step,
                            lagtimes,
                            k,
                            save=False
                           ):
    its_trajs = []
    its_trajs_u = []
    its_trajs_rw = []
    for t in range(n_trajs):
        file_name = '2D_'+str(t) + '_' + str(100)
        if shared_discretisation:
           its_trajs.append(np.loadtxt(store_directory + implied_timescales +'mp_'+ file_name + ".dat"))
           its_trajs_rw.append(np.loadtxt(store_directory + implied_timescales +'rw_mp_'+file_name+'.dat'))
           its_trajs_u.append(np.loadtxt('nobackup/' + implied_timescales +'mp_'+ file_name + ".dat"))
        else:
            its_trajs.append(np.loadtxt(store_directory + implied_timescales + file_name + ".dat"))
            its_trajs_rw.append(np.loadtxt(store_directory + implied_timescales +'rw_'+file_name+'.dat'))
            its_trajs_u.append(np.loadtxt('nobackup/' + implied_timescales + file_name + ".dat"))

    its_data = np.array(its_trajs) 
    its_data_rw = np.array(its_trajs_rw) 
    its_data_u = np.array(its_trajs_u) 

    mean = np.mean(its_data, axis=0) * time_step
    std  = np.std(its_data, axis=0) * time_step

    mean_rw = np.mean(its_data_rw, axis=0) * time_step
    std_rw  = np.std(its_data_rw, axis=0) * time_step

    mean_u = np.mean(its_data_u, axis=0) *time_step
    std_u  = np.std(its_data_u, axis=0) *time_step

    lagtimes = np.array(lagtimes) * time_step

    fig, ax = plt.subplots(1,1, figsize=(15,10))
    #fig.suptitle('Implied timescales',x=0.53,y=0.9,fontsize=17)
    for ev in range(k-1):
        ax.plot(lagtimes, mean[:,ev] ,':',  c='C'+str(ev), 
                    lw=3,label=str(ev+1)+'. biased process')
        ax.fill_between(lagtimes,mean[:,ev]-std[:,ev],
                            mean[:,ev]+std[:,ev], 
                            color='C'+str(ev),alpha=.1)
        
        ax.plot(lagtimes, mean_rw[:,ev] ,  c='C'+str(ev),  #[:,ev]
                    lw=1.5,label=str(ev+1)+'. reweighted process')
        ax.fill_between(lagtimes,mean_rw[:,ev]-std_rw[:,ev],
                            mean_rw[:,ev]+std_rw[:,ev], 
                            color='C'+str(ev),alpha=.1)
        
        ax.plot(lagtimes, mean_u[:,ev] ,'--',  c='C'+str(ev), 
                    lw=3,label=str(ev+1)+'. unbiased process')
        ax.fill_between(lagtimes,mean_u[:,ev]-std_u[:,ev],
                            mean_u[:,ev]+std_u[:,ev], 
                            color='C'+str(ev),alpha=.1)
               
    ax.plot(lagtimes,np.array(lagtimes)*2, color='C7', lw=0.4)
    H = np.array(lagtimes)*2
    d = np.zeros(len(lagtimes))
    ax.fill_between(lagtimes, H, where=H>=d, interpolate=True, 
                            color='C7', alpha=0.1)
        
        
        
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_ylabel('timescale in ps', fontsize=17, labelpad=15)
    ax.set_xlabel('lag time in ps', fontsize=17, labelpad=15)
    ax.legend( fontsize=17,
              bbox_to_anchor=(0, 1, 1, 0), 
              loc="lower left", 
              mode="expand", 
              ncol=3)
    
    #title='Implied timescales', title_fontsize=20,
                      #loc='lower center', bbox_to_anchor=(1, 0.25), fontsize=17)
     # plt.tight_layout()#pad=1, w_pad=0., h_pad=0.)    
    if save:                                 
        fig.savefig(store_directory + implied_timescales + 'its_rw.png' , bbox_inches='tight', format='png')     




def plot_difference_plot(ABD, SBD, DP, grid_s, t, tau,
                         store_directory,
                         trajectories,
                         differenc_plots,
                         scaling=0.025,
                         name='DP_KL_',
                         save=False
        ):
        ## Kullback-Leibler divergence
        SABD = np.divide(SBD, ABD, 
                         out=np.zeros(SBD.shape, dtype=float),
                         where=ABD!=0)
        log_SABD = res = np.log2(SABD, 
                                 out=np.zeros_like(SABD), 
                                 where=(SABD!=0))
        KLD = SBD * log_SABD
        ## for x-dim
        Dx = (KLD).sum(axis=0)
        ## for y-dim 
        Dy = (KLD).sum(axis=1)
        
        KLD = KLD.sum()

        x = Dx
        y = Dy

        # definitions for the axes
        left, width    = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing        = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histx   = [left, bottom + height + spacing, width, 0.2]
        rect_histy   = [left + width + spacing, bottom, 0.2, height]

        # start with a rectangular Figure
        fig = plt.figure(figsize=(9, 11.5))

        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(direction='in', top=True, right=True)
        ax_histx   = plt.axes(rect_histx, sharex=ax_scatter)
        ax_histx.tick_params(direction='in', labelbottom=False)
        ax_histy   = plt.axes(rect_histy, sharey=ax_scatter)
        ax_histy.tick_params(direction='in', labelleft=False)

        bound = max(abs(DP.min()),abs(DP.max()))
        norm=colors.SymLogNorm(linthresh=bound*scaling, linscale=1,
                               vmin=-bound, vmax=bound, base=10)
        #colors.CenteredNorm()

        pos = ax_scatter.pcolor(grid_s[0], grid_s[1], 
                                DP, shading='auto', 
                                norm=norm,
                                cmap='seismic')

        #bins = np.arange(-lim, lim + binwidth, binwidth)
        #ax_histx.plot(grid_s[0],x,color='dimgrey', lw=2.5)
        ax_histx.fill_between(grid_s[0], x, where=x>=0, interpolate=True,
                              color='red',alpha=.85)
        ax_histx.fill_between(grid_s[0], x, where=x<=0, interpolate=True, 
                              color='blue',alpha=.7)


        #ax_histy.plot(y,grid_s[1],color='dimgrey') 
        #ax_histy.fill_between(y,grid_s[1],color='red',alpha=.1)
        ax_histy.fill_betweenx(grid_s[1], y, where=y>=0, interpolate=True, 
                               color='red',alpha=.85)
        ax_histy.fill_betweenx(grid_s[1], y, where=y<=0, interpolate=True, 
                               color='blue',alpha=.7)

        ax_histx.set_xlim(ax_scatter.get_xlim())
        ax_histy.set_ylim(ax_scatter.get_ylim())

        #ax_histx.set_title(r'Difference Plot', fontsize=25)
        ax_histx.grid(True, axis='x')
        ax_histy.grid(True, axis='y')
        ax_scatter.set_xlabel(r'x',fontsize=23)
        #ax_scatter.yticks(fontsize=10)
        ax_scatter.set_ylabel(r'y',fontsize=23)
        ax_scatter.grid(True)

        ax_scatter.tick_params(axis='both', which='major', labelsize=23)
        ax_histy.tick_params(axis='both', rotation=25, which='major', labelsize=23)
        ax_histx.tick_params(axis='both', which='major', labelsize=23)
        #ax_histy.set_xscale('log')
        def y_fmt(x,y):
            return '${:1.0e}'.format(x).replace('e', '\cdot 10^{') + '}$'

        ax_histx.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
        ax_histy.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
        
        #import matplotlib.ticker

        #logfmt = matplotlib.ticker.LogFormatterExponent(base=10.0)
        #ax_histx.yaxis.set_major_formatter(logfmt)
        #ax_histy.xaxis.set_major_formatter(logfmt)

            
        cbar = fig.colorbar(pos, ax=[ax_scatter,ax_histy], orientation= 'horizontal')
        #cbar.ax.tick_params(labelsize=50)
        #pos.locator = tick_locator
        #pos.update_ticks()
        cbar.ax.tick_params(labelsize=23, grid_alpha=0.5, direction='in')
        #cbar.set_label(label='',fontsize=25,  weight='light')
        
        if save:
            plt.savefig(store_directory+differenc_plots+name+str(t)+'_'+str(tau)+'.png' , bbox_inches='tight', format='png')
        return t, tau, KLD

def plot_biased_unbiased_potential(unbiased,
                                   biased,
                                   grid,
                                   vmax,
                                   name,
                                   directory,
                                   save=False
                                  ):
    
    P2D = unbiased(grid[0], grid[1])
    P2DB = biased(grid[0], grid[1])
    A=[P2D,P2DB]

    fig, a = plt.subplots(1,2, figsize=(25,10), sharex=True, sharey=True)
    a = a.ravel()
    g=[]
    for idx, ax in enumerate(a):
        if idx==0:
            ax.set_ylabel('y', fontsize=20, labelpad=15)
            
        pos=ax.pcolor(grid[0], grid[1], A[idx], shading='auto', 
                      vmax=vmax, cmap='tab20c')
        g.append(ax)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel('x', fontsize=20, labelpad=15)
    cbar=fig.colorbar(pos, ax=g, location='right', anchor=(-0.1, 0.3))
    cbar.ax.tick_params(labelsize=20) 
    
    if save:
        fig.savefig(directory + name +'.png' , bbox_inches='tight', 
                    format='png')      


def plot_3_2D_eigenvectors(vecs, 
                           lcs,
                           tau, 
                           t, 
                           minv,
                           maxv,
                           scaling,
                           directory,
                           save=False
                          ):
    
    grid = np.linspace(minv,maxv,tau+1).squeeze()
    grid = (grid[1:]+grid[:-1])/2
    
    vmin = vecs.min().real
    vmax = vecs.max().real
    
    fig, axes = plt.subplots(1, 3, figsize=(28, 11.5), sharex=True, sharey=True)
    a=[]
    for ax, i in zip(axes.flat, range(1,vecs.shape[0])): 
        Vec=np.zeros(tau*tau)
        Vec[lcs]=vecs[i,:]
        Vec = Vec.reshape(tau,tau)
        H=Vec #.reshape(100,100)
        
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        #ax.grid(True, axis='both', color='k')
        
        if i == 1:
            ax.set_ylabel('y', fontsize=30, labelpad=15)
        else:
            a.append(ax)
        ax.set_xlabel('x', fontsize=30, labelpad=15)
          
         
            
        bound = max(abs(vmin),abs(vmax))
        norm=colors.SymLogNorm(linthresh=bound*scaling, linscale=1,
                               vmin=-bound, vmax=bound, base=10)
        im = ax.pcolor(grid[:,0], grid[:,1], H, shading='auto', 
                       norm=norm, cmap='coolwarm')
            

        
    cbar = fig.colorbar(im,ax=axes,  orientation= 'horizontal')
    cbar.ax.tick_params(labelsize=30, grid_alpha=0.2, direction='in')
    if save:
        plt.savefig(directory+'eigenvectors_3_2D_'+str(t)+'_'+str(tau)+'.png' ,
                    bbox_inches='tight', format='png')
       
    
def plot_2D_2D1D_1D_adt_eigenvectors(signx,
                                     signy, 
                                     ev,
                                     vecs2D,
                                     lcs2D,
                                     vecs1DX,
                                     lcs1DX,
                                     vecs1DY,
                                     lcs1DY, 
                                     tau, 
                                     minv,
                                     maxv,
                                     scaling,
                                     store_directory, 
                                     dynamics,
                                     save=False
                                ):
    
    grid = np.linspace(minv,maxv,tau+1).squeeze()
    grid = (grid[1:]+grid[:-1])/2
            
    vmin = vecs2D.min().real
    vmax = vecs2D.max().real
    bound = max(abs(vmin),abs(vmax))
    
    Vec=np.zeros(tau*tau)
    #if max(lcs2D)>tau*tau:
      #  l=len(lcs2D[lcs2D>tau*tau])
     #   Vec[lcs2D[:-l]]=vecs2D[ev,:-l]
            
    #else:
    Vec[lcs2D]=vecs2D[ev,:]
    Vec = Vec.reshape(tau,tau)
    H=Vec 
    
    Vecs1DX=np.zeros(tau)
    vecs1DX[ev,:]=signx*vecs1DX[ev,:]
    Vecs1DX[lcs1DX]=vecs1DX[ev,:]
    
    Vecs1DY=np.zeros(tau)
    vecs1DY[ev,:]=signy*vecs1DY[ev,:]
    Vecs1DY[lcs1DY]=vecs1DY[ev,:]

    ## 2D -> 1D Projection
    x = Vec.sum(axis=0)
    y = Vec.sum(axis=1)
            
                
    # definitions for the axes
    left, width    = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing        = 0.005
            
    rect_scatter = [left, bottom, width, height]
    rect_histx   = [left, bottom + height + spacing, width, 0.2]
    rect_histy   = [left + width + spacing, bottom, 0.2, height]
            
    # start with a rectangular Figure
    fig = plt.figure(figsize=(9, 11))
            
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx   = plt.axes(rect_histx, sharex=ax_scatter)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy   = plt.axes(rect_histy, sharey=ax_scatter)
    ax_histy.tick_params(direction='in', labelleft=False)
                
    bound = max(abs(vmin),abs(vmax))
    norm=colors.SymLogNorm(linthresh=bound*scaling, linscale=1,
                           vmin=-bound, vmax=bound, base=10)
    im = ax_scatter.pcolor(grid[:,0], grid[:,1], H, shading='auto', 
                           norm=norm, cmap='coolwarm')
            
    
    ax_histx.plot(grid[:,0],Vecs1DX,color='dimgrey', lw=2.5)
    ax_histx.fill_between(grid[:,0], x, where=x>=0, interpolate=True,
                              color='coral',alpha=.78)
    ax_histx.fill_between(grid[:,0], x, where=x<=0, interpolate=True, 
                                  color='cornflowerblue',alpha=.8)
            
            
    
    ax_histy.plot(Vecs1DY,grid[:,1],color='dimgrey', lw=2.5)
    ax_histy.fill_betweenx(grid[:,1], y, where=y>=0, interpolate=True, 
                                   color='coral',alpha=.78)
    ax_histy.fill_betweenx(grid[:,1], y, where=y<=0, interpolate=True, 
                                   color='cornflowerblue',alpha=.8)
            
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
            
    #ax_histx.set_title(r'Difference Plot', fontsize=25)
    ax_histx.grid(True, axis='x')
    ax_histy.grid(True, axis='y')
    ax_scatter.set_xlabel(r'x',fontsize=20)
    #ax_scatter.yticks(fontsize=10)
    ax_scatter.set_ylabel(r'y',fontsize=20)
    ax_scatter.grid(True)
    #ax_histy.set_facecolor('gainsboro')
    #ax_histx.set_facecolor('gainsboro')
            
    ax_scatter.tick_params(axis='both', which='major', labelsize=20)
    ax_histy.tick_params(axis='both', rotation=35, which='major', labelsize=20)
    ax_histx.tick_params(axis='both', which='major', labelsize=20)
    #def y_fmt(x,y):
        #return '${:1.0e}'.format(x).replace('e', '\cdot 10^{') + '}$'
                
    #ax_histx.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
    #ax_histy.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
        
    cbar = fig.colorbar(im,ax=[ax_scatter,ax_histy],  orientation= 'horizontal')
    cbar.ax.tick_params(labelsize=20, rotation=35, grid_alpha=0.2, direction='in')
            
    if save:        
        plt.savefig(store_directory+dynamics+'adt_2D_2D1D_1D_eigenvectors_'
            +str(ev)+'_'+str(tau)+'.png' , bbox_inches='tight', format='png')   
   

 ## do not use but keep to see funktionality      
def plot_save_2D_eigenvecs_1D_projection(vecs,vecs1DX,vecs1DY, lcs,tau, t, grid_s, 
                                    store_directory, dynamics,
                                    directory, single=False, save=False):
    vmin = vecs.min().real
    vmax = vecs.max().real
    
    fig, axes = plt.subplots(1, 3, figsize=(28, 11.5), sharex=True, sharey=True)
    a=[]
    X2D1D=[]
    Y2D1D=[]
    for ax, i in zip(axes.flat, range(1,vecs.shape[0])): 
        Vec=np.zeros(tau*tau)
        Vec[lcs]=vecs[i,:]
        Vec = Vec.reshape(tau,tau)
        H=Vec #.reshape(100,100)
        
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.grid()
        
        if single is False:
            #if i % 2 == 0:
            if i == 1:
                ax.set_ylabel('y', fontsize=30, labelpad=15)
            else:
                a.append(ax)
            #if i == unbiased_eigenvecs.shape[0]-2 or i == unbiased_eigenvecs.shape[0]-1:
            ax.set_xlabel('x', fontsize=30, labelpad=15)
          
         
            
            bound = max(abs(vmin),abs(vmax))
            norm=colors.SymLogNorm(linthresh=bound*0.01, linscale=1,
                                                              vmin=-bound, vmax=bound, base=10)
            #np.linspace(traj[:,0].min(),traj[:,0].max(),H.shape[0]), 
            #                   np.linspace(traj[:,1].min(),traj[:,1].max(),H.shape[1]),
            im = ax.pcolor(grid_s[0], grid_s[1], H, shading='auto', 
                                 norm=norm,
                                 cmap='coolwarm')
            

        elif single is True:
            ax.set_ylabel('y', fontsize=30, labelpad=15)
            ax.set_xlabel('x', fontsize=30, labelpad=15)

            ## 2D -> 1D Projection
            x = Vec.sum(axis=0)
            X2D1D.append(x)
            y = Vec.sum(axis=1)
            Y2D1D.append(y)
            
                
            # definitions for the axes
            left, width    = 0.1, 0.65
            bottom, height = 0.1, 0.65
            spacing        = 0.005
            
            rect_scatter = [left, bottom, width, height]
            rect_histx   = [left, bottom + height + spacing, width, 0.2]
            rect_histy   = [left + width + spacing, bottom, 0.2, height]
            
            # start with a rectangular Figure
            fig = plt.figure(figsize=(9, 11))
            
            ax_scatter = plt.axes(rect_scatter)
            ax_scatter.tick_params(direction='in', top=True, right=True)
            ax_histx   = plt.axes(rect_histx, sharex=ax_scatter)
            ax_histx.tick_params(direction='in', labelbottom=False)
            ax_histy   = plt.axes(rect_histy, sharey=ax_scatter)
            ax_histy.tick_params(direction='in', labelleft=False)
                
            bound = max(abs(vmin),abs(vmax))
            norm=colors.SymLogNorm(linthresh=bound*0.04, linscale=1,
                                   vmin=-bound, vmax=bound, base=10)
            #np.linspace(traj[:,0].min(),traj[:,0].max(),H.shape[0]), 
            #                   np.linspace(traj[:,1].min(),traj[:,1].max(),H.shape[1]),
            im = ax_scatter.pcolor(grid_s[0], grid_s[1], H, shading='auto', 
                                   norm=norm,
                                   cmap='coolwarm')
            
            #bins = np.arange(-lim, lim + binwidth, binwidth)
            ax_histx.plot(grid_s[0],vecs1DX[i,:],color='dimgrey', lw=2.5)
            ax_histx.fill_between(grid_s[0], x, where=x>=0, interpolate=True,
                                  color='coral',alpha=.78)
            ax_histx.fill_between(grid_s[0], x, where=x<=0, interpolate=True, 
                                  color='cornflowerblue',alpha=.8)
            
            
            
            #ax_histy.plot(y,grid_s[1],color='dimgrey') 
            #ax_histy.fill_between(y,grid_s[1],color='red',alpha=.1)
            ax_histy.plot(vecs1DY[i,:],grid_s[1],color='dimgrey', lw=2.5)
            ax_histy.fill_betweenx(grid_s[1], y, where=y>=0, interpolate=True, 
                                   color='coral',alpha=.78)
            ax_histy.fill_betweenx(grid_s[1], y, where=y<=0, interpolate=True, 
                                   color='cornflowerblue',alpha=.8)
            
            ax_histx.set_xlim(ax_scatter.get_xlim())
            ax_histy.set_ylim(ax_scatter.get_ylim())
            
            #ax_histx.set_title(r'Difference Plot', fontsize=25)
            ax_histx.grid(True, axis='x')
            ax_histy.grid(True, axis='y')
            ax_scatter.set_xlabel(r'y',fontsize=16)
            #ax_scatter.yticks(fontsize=10)
            ax_scatter.set_ylabel(r'x',fontsize=16)
            ax_scatter.grid(True)
            #ax_histy.set_facecolor('gainsboro')
            #ax_histx.set_facecolor('gainsboro')
            
            ax_scatter.tick_params(axis='both', which='major', labelsize=16)
            ax_histy.tick_params(axis='both', rotation=35, which='major', labelsize=16)
            ax_histx.tick_params(axis='both', which='major', labelsize=16)
            #def y_fmt(x,y):
                #return '${:1.0e}'.format(x).replace('e', '\cdot 10^{') + '}$'
                
            #ax_histx.yaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
            #ax_histy.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
        
            cbar = fig.colorbar(im,ax=[ax_scatter,ax_histy],  orientation= 'horizontal')
            cbar.ax.tick_params(labelsize=30, rotation=35, grid_alpha=0.2, direction='in')
            
            
            if save:
                plt.savefig(directory+'eigenvector_2D1D_'+str(i)+'_'
                            +str(t)+'_'+str(tau)+'.png' , 
                            bbox_inches='tight', format='png')
   
    np.save(store_directory+dynamics+'eigenvectorsX_2Dto1D_'
            +str(t)+'_'+str(tau), np.array(X2D1D))
    np.save(store_directory+dynamics+'eigenvectorsY_2Dto1D_'
            +str(t)+'_'+str(tau), np.array(Y2D1D))
    
    if single is False:        
        cbar = fig.colorbar(im,ax=axes,  orientation= 'horizontal')
        cbar.ax.tick_params(labelsize=30, grid_alpha=0.2, direction='in')
        if save:
            plt.savefig(directory+'eigenvector_2D_cbar_'+str(t)+'_'+str(tau)+'.png' ,
                            bbox_inches='tight', format='png')
       
    #fig.colorbar(im,ax=axes, pad= 0.02, orientation='horizontal', shrink=0.65)
    #fig.subplots_adjust(bottom=0.8)

        

    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #cbar = fig.colorbar(im1, cax=cbar_ax)
    #cbar = fig.colorbar(im1,ax=a, orientation= 'vertical')
    #cbar.ax.tick_params(labelsize=30, grid_alpha=1, direction='in')

        
def plot_2D_counterplot_like(meshgrid, 
                             Distribution, 
                             directory,
                             name='eigenvector',
                             save=False
                             ):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    
    im = ax.pcolor(meshgrid[0], meshgrid[1], Distribution, 
                   shading='auto', 
                   cmap='tab20c')

    cbar = fig.colorbar(im,ax=ax,  orientation= 'horizontal')
    cbar.ax.tick_params(labelsize=30, grid_alpha=0.2, direction='in')
    
    ax.set_ylabel('y', fontsize=30, labelpad=15)
    ax.set_xlabel('x', fontsize=30, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.grid()
    if save:
        plt.savefig(directory+name+'.png' , bbox_inches='tight', format='png')