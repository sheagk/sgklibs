#!/usr/bin/env python3

def convert_to_per_pc2(counts, bsize_in_kpc):
    '''
    converts a 2d histogram of mass per bin-size^2
    to mass per pc squared, assuming that the bin
    size is in kpc 

    '''
    binlen_pc = bsize_in_kpc*1e3
    bin_area = binlen_pc**2
    #if I have,e.g. 10 Msun/(150 pc * 150 pc), then I also have (10/(150*150)) Msun / pc^2 => divide by the bin area (in pc) to get per pc 
    conversion_factor = 1./bin_area
    return counts * conversion_factor


def viz_part_indices(hal_i,part,hal,species_name='star',kind='3panel',
                   bsize=20,nbins=100,cvmin=1e-3,cvmax=1e3,
                   ided_part_method='hist',ided_kwargs={'alpha':1.0,'cmap':'cubehelix'},
                   nearby_part_method = 'scatter',nearby_kwargs={'s':1, 'edgecolor':'None','facecolor':'cyan', 'marker':','},
                   figsize=(13,13), fontsize=28,label=None,
                   rotate=False,rotate_cut=0.5,rot_vec=None,return_rv=False,
                   circcolor='w',circprop='star.radius.90',circmult=1.,
                   cbar=True,verbose=False,
                   center_prop='star.position',
                   recenter=False,
                   add_to_label=['host.distance.total']):
    '''
    visualize particles belonging to a halo/galaxy and nearby particles not belonging 
    to that halo/galaxy.  either can be plotted as either a 2D histogram or a set of 
    scatterpoints; the _kwargs are passed to the appropriate plotting command.

    Parameters
    ----------

    hal_i : int : index of the halo that you're visualizing
    part : particle dictionary : particle data with particle species to visualize
    hal : halo catalog : halo catalog that contains a spec+'.indices' dataset for index hal_i
    species_name : string : partticle species to visualize
    kind :string : 
        one of ['3panel', 'faceon', 'edgeon'].  3 panel displays all 3 projections in a
        2x2 grid.  faceon and edgeon show just one view; if rotate is true, then they show
        that visualization; if false, you'll just get two different (but random) projections
    bsize : float : half size of the box to visualize (comoving kpc)
    nbins : int : number of bins to use for 2dhistograms
    cvmin : float : minimum of the color normalization for histograms
    cvmax : float : max of the color normalizations
    ided_part_method : string : 
        either 'hist' or 'scatter' to do either 2d histograms or scatter plots 
        of the particles associated with this object (according to hal)
    ided_kwargs : dictionary : kwargs passed to either ax.scatter or ax.pcolormesh
    nearby_part_method : string : 
        either 'hist' or 'scatter' to do either 2d histograms or scatter plots
        of partciles in the box but not associated with the targeted object
    nearby_kwargs : dictionary : kwargs passed to either ax.scatter or ax.pcolormesh
    figsize : tuple : figure size for matplotlib
    fontsize : float : fontsize for matplot
    label : string : text added to the empty area in the 2x2 plot.  not shown if 
        face-on or edge-on image
    rotate : boolean : whether to rotate the images to align with principal axes
    rotate_cut : float : fraction of the box size to use to calculate rotation vectors
    rot_vec : array : rotation vectors to use
    return_rv : boolean : whether to return the calculated rotation vectors
    circcolor : string : color of the circle to put around the halo
    circprop : string : what property to use for the circle radius
    circmult : float : what to multiple the circle radius by
    cbar : boolean : whether or not ot do a colorbar in the empty axes, or alongsize
    verbose : boolean : whether or not to print extra info
    recenter: boolean : whether or not to recenter on the particle indices
    add_to_label : list : 
        properties to add their value to the label.  default box.fraction is calculated
        by this program, but all others must be in hal
    '''

    assert kind in ['3panel','edgeon','faceon']

    import numpy as np    
    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec


    import utilities as ut
    from .low_level_utils import fast_dist, boxmsk

    part_indices = hal[species_name+'.indices'][hal_i]

    # center_position = np.r_[-1,-1,-1]
    # if 'star.position' in hal:
    #     center_position = hal['star.position'][hal_i]
    # if center_position[0] <= 0 or 'star.position' not in hal:
    #     center_position = hal['position'][hal_i]
    center_position = hal[center_prop][hal_i]

    if recenter:
        old_pos = np.array(center_position,copy=True)
        center_position = ut.coordinate.get_position_center_of_mass_zoom(part[species_name]['position'][part_indices], 
            part[species_name]['mass'][part_indices], part.info['box.length'],position_number_min=8)
        print("Center moved by {} kpc".format(fast_dist(old_pos,center_position)))

    radius = hal[circprop][hal_i]
    radius_to_plot = radius*circmult
    
    if rotate:
        if rot_vec is None:
            RV,EV,AR = ut.particle.get_principal_axes(part,species=species_name,center_position=center_position,distance_max=bsize*rotate_cut)
        else:
            RV = rot_vec
        rot_pos = ut.particle.get_distances_wrt_center(part,species=species_name,center_position=center_position,distance_kind='vector',rotation=RV)
        particles_in_box = boxmsk(rot_pos,[0,0,0],bsize) #already shifted
    else:
        rot_pos = part[species_name]['position'] - center_position
        particles_in_box = boxmsk(part[species_name]['position'],center_position,bsize)   

    nbox = np.count_nonzero(particles_in_box)

    my_msk = np.empty_like(particles_in_box)
    my_msk[:] = False
    my_msk[part_indices] = True
    
    #particles assigned to this halo that are in the box:
    selected_msk = my_msk & particles_in_box
    nmine = np.count_nonzero(selected_msk)
    
    #particles that are assigned to this halo that are outside the box:
    missed_msk = my_msk & np.logical_not(particles_in_box)    
    nmissed = np.count_nonzero(missed_msk)
    
    #particles not assigned to this halo that are inside the box:
    unselected_msk = np.logical_not(my_msk) & particles_in_box
    nother = np.count_nonzero(unselected_msk)
    
    if verbose:
        print("Plotting particles in a cube of half side length = {} kpc; {} = {:.1f} kpc"
              .format(bsize,circprop,radius))
        print("Have {} total particles in the box".format(np.count_nonzero(particles_in_box)))
        print("{} of those ({:.1f}%) are associated with the halo"
              .format(nmine,nmine*100./nbox))
        print("{} of the particles associated with the halo ({:.1f}%) are outside the box"
              .format(nmissed,nmissed*100./part_indices.shape[0]))          
        
    my_pos = rot_pos[selected_msk]
    my_weights = part[species_name]['mass'][selected_msk]
    
    other_pos = rot_pos[unselected_msk]
    other_weights = part[species_name]['mass'][unselected_msk]   
    
    if ided_part_method.startswith('hist'):
        ided_part_method = 'hist'
    elif ided_part_method.startswith('scatter'):
        ided_part_method = 'scatter'

    if nearby_part_method.startswith('hist'):
        nearby_part_method = 'hist'
    elif nearby_part_method.startswith('scatter'):
        nearby_part_method = 'scatter'
        
    lim = [int(round(-bsize)),int(round(bsize))]
    tickloc = []
    tickstr = []

    if ided_part_method == 'hist' and 'cmap' in ided_kwargs:
        cmap = matplotlib.cm.get_cmap(ided_kwargs['cmap'])
        bgcolor = list(cmap(0)[:-1])
        scalecolor = 'w'
    elif nearby_part_method and 'cmap' in nearby_kwargs:
        cmap = matplotlib.cm.get_cmap(nearby_kwargs['cmap'])
        bgcolor = list(cmap(0)[:-1])
        scalecolor = 'w'
    else:
        cbar = False #can't do a colorbar without a colormap
        bgcolor = 'w'
        scalecolor = 'k'
    cNorm = LogNorm(vmin=cvmin,vmax=cvmax)

    fig = plt.figure(figsize=figsize)
    if kind == '3panel':
        ax1 = fig.add_subplot(221,facecolor=bgcolor,aspect='equal')   #x vs z
        ax2 = fig.add_subplot(222,facecolor='w')   #nothing; maybe a label
        ax3 = fig.add_subplot(223,facecolor=bgcolor,sharex=ax1,aspect='equal')   #x vs y -- shares the xaxis with the plot above it
        ax4 = fig.add_subplot(224,facecolor=bgcolor,sharey=ax3,aspect='equal')   #z vs y -- shares the y axis with the plot beside it

        ax3.set_xlabel(r'$x$', fontsize=fontsize)
        ax3.set_ylabel(r'$y$', fontsize=fontsize)
        ax4.set_xlabel(r'$z$', fontsize=fontsize)
        ax1.set_ylabel(r'$z$', fontsize=fontsize)
    else:
        gs = GridSpec(1,2,width_ratios=[12,1],wspace=0.02)
        ax = plt.subplot(gs[0],facecolor=bgcolor,aspect='equal')
        cax = plt.subplot(gs[1])

    xp,yp,zp = my_pos.T
    if ided_part_method == 'hist':   
        if kind == '3panel':
            #first, x vs z in ax1
            hist,xe,ye = np.histogram2d(xp,zp,bins=nbins,weights=my_weights)
            hist = convert_to_per_pc2(hist,xe[1]-xe[0])
            Y_eo,X_eo = np.meshgrid(xe,ye)
            im = ax1.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**ided_kwargs) 

            #second, x vs y in ax3
            hist,xe,ye = np.histogram2d(xp,yp,bins=nbins,weights=my_weights)
            hist = convert_to_per_pc2(hist,xe[1]-xe[0])
            Y_eo,X_eo = np.meshgrid(xe,ye)
            im = ax3.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**ided_kwargs) 

            #last, z vs y in ax4
            hist,xe,ye = np.histogram2d(zp,yp,bins=nbins,weights=my_weights)
            hist = convert_to_per_pc2(hist,xe[1]-xe[0])
            Y_eo,X_eo = np.meshgrid(xe,ye)
            im = ax4.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**ided_kwargs) 
        else:
            if kind == 'edgeon':
                #want x vs z:
                xi = xp
                yi = zp    
            elif kind == 'faceon':
                #x vs y
                xi = xp
                yi = yp
            hist,xe,ye = np.histogram2d(xi,yi,bins=nbins,weights=my_weights)
            hist = convert_to_per_pc2(hist,xe[1]-xe[0])
            Y_eo,X_eo = np.meshgrid(xe,ye)
            im = ax.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**ided_kwargs)
            if cbar:
                cbar = plt.colorbar(im,cax=cax)
                cbar.set_label(r'projected density [$M_\odot$/pc$^2$]', labelpad=15)

    elif ided_part_method == 'scatter':   
        if kind == '3panel':
            #first, x vs z in ax1
            ax1.scatter(xp,zp,**ided_kwargs)

            #second, x vs y in ax3
            ax3.scatter(xp,yp,**ided_kwargs)

            #last, z vs y in ax4
            ax4.scatter(zp,yp,**ided_kwargs)
        else:
            if kind == 'edgeon':
                #want x vs z:
                xi = xp
                yi = zp    
            elif kind == 'faceon':
                #x vs y
                xi = xp
                yi = yp
            ax.scatter(xi,yi,**ided_kwargs)                
    else:
        print("!! don't know method for plotting particles as {}; not plotting selected particles".format(mine_method))

    xp,yp,zp = other_pos.T
    if nearby_part_method == 'hist':  
        if kind == '3panel': 
            #first, x vs z in ax1
            hist,xe,ye = np.histogram2d(xp,zp,bins=nbins,weights=other_weights)
            hist = convert_to_per_pc2(hist,xe[1]-xe[0])
            Y_eo,X_eo = np.meshgrid(xe,ye)
            im = ax1.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**nearby_kwargs) 

            #second, x vs y in ax3
            hist,xe,ye = np.histogram2d(xp,yp,bins=nbins,weights=other_weights)
            hist = convert_to_per_pc2(hist,xe[1]-xe[0])
            Y_eo,X_eo = np.meshgrid(xe,ye)
            im = ax3.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**nearby_kwargs) 

            #last, z vs y in ax4
            hist,xe,ye = np.histogram2d(zp,yp,bins=nbins,weights=other_weights)
            hist = convert_to_per_pc2(hist,xe[1]-xe[0])
            Y_eo,X_eo = np.meshgrid(xe,ye)
            im = ax4.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**nearby_kwargs)
        else:
            if kind == 'edgeon':
                #want x vs z:
                xi = xp
                yi = zp    
            elif kind == 'faceon':
                #x vs y
                xi = xp
                yi = yp
            hist,xe,ye = np.histogram2d(xi,yi,bins=nbins,weights=my_weights)
            hist = convert_to_per_pc2(hist,xe[1]-xe[0])
            Y_eo,X_eo = np.meshgrid(xe,ye)
            im = ax.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**nearby_kwargs)
            if cbar and ided_part_method != 'hist':
                cbar = plt.colorbar(im,cax=cax)              
    
    elif nearby_part_method == 'scatter':   
        if kind == '3panel':
            #first, x vs z in ax1
            ax1.scatter(xp,zp,**nearby_kwargs)

            #second, x vs y in ax3
            ax3.scatter(xp,yp,**nearby_kwargs)

            #last, z vs y in ax4
            ax4.scatter(zp,yp,**nearby_kwargs)
        else:
            if kind == 'edgeon':
                #want x vs z:
                xi = xp
                yi = zp    
            elif kind == 'faceon':
                #x vs y
                xi = xp
                yi = yp
            ax.scatter(xi,yi,**nearby_kwargs)  
    elif nearby_part_method in ['none','None',None]:
        pass
    else:
        print("!! don't know method for plotting particles as {}; not plotting nearby particles".format(other_method))


    width = 2*bsize
    if width > 1000:
        scalesize = 300
        scalelabel = r'$\mathrm{300\,kpc}$'
    elif width < 1000 and width > 500:
        scalesize = 100
        scalelabel = r'$\mathrm{100\,kpc}$'
    elif width < 600 and width > 100:
        scalesize = 50
        scalelabel = r'$\mathrm{50\,kpc}$'
    elif width > 50 and width < 100:
        scalesize = 10
        scalelabel = r'$\mathrm{10\,kpc}$'
    elif width <= 50 and width > 9:
        scalesize = 5
        scalelabel = r'$\mathrm{5\,kpc}$'
    elif width < 10 and width > 2:
        scalesize = 1
        scalelabel = r'$\mathrm{1\,kpc}$'
    else:
        scalesize = 0.1
        scalelabel = r'$\mathrm{100\,pc}$'


    x1 = - bsize + width/20
    x2 = x1+scalesize
    y = width/20 - bsize

    if kind == '3panel':
        ax1.set_xlim(lim)
        ax1.set_ylim(lim)

        ax2.set_xlim(lim)
        ax2.set_ylim(lim)

        ax3.set_xlim(lim)
        ax3.set_ylim(lim)

        ax4.set_xlim(lim)
        ax4.set_ylim(lim)

        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.setp(ax1.get_yticklabels(),visible=False)
        plt.setp(ax1.get_xticklabels(),visible=False)

        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.setp(ax2.get_yticklabels(),visible=False)
        plt.setp(ax2.get_xticklabels(),visible=False)

        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.setp(ax3.get_yticklabels(),visible=False)
        plt.setp(ax3.get_xticklabels(),visible=False)

        ax4.set_xticks([])
        ax4.set_yticks([])
        plt.setp(ax4.get_yticklabels(),visible=False)
        plt.setp(ax4.get_xticklabels(),visible=False)

        fig.subplots_adjust(left=0.05,bottom=0.05,top=0.95,right=0.95,hspace=0.025,wspace=0.025)

        ax1.plot([x1,x2],[y,y],lw=3,ls='-',color=scalecolor,label='_nolegend_')
        ax1.text((x1+x2)/2.,y+width/50,scalelabel,color=scalecolor,ha='center',va='bottom',fontsize=0.75*fontsize)

        ax3.plot([x1,x2],[y,y],lw=3,ls='-',color=scalecolor,label='_nolegend_')
        ax3.text((x1+x2)/2.,y+width/50,scalelabel,color=scalecolor,ha='center',va='bottom',fontsize=0.75*fontsize)

        ax4.plot([x1,x2],[y,y],lw=3,ls='-',color=scalecolor,label='_nolegend_')
        ax4.text((x1+x2)/2.,y+width/50,scalelabel,color=scalecolor,ha='center',va='bottom',fontsize=0.75*fontsize)

        if label is None and add_to_label != []:
            label = ''
        for prop in add_to_label:
            if prop == 'box.fraction':
                label += '\nbox fraction = {:.2g}'.format(nmine*1.0/nbox)
            else:
                try:
                    pval = hal.prop(prop)[hal_i]
                    label += '\n'+prop+' = {:.3g}'.format(pval)
                except Exception:
                    pass

        if label is not None:
            ax2.text(0.5,0.1,label,ha='center',va='bottom',transform=ax2.transAxes,color='k',fontsize=fontsize)

        if cbar:
            cax,kw = mpl.colorbar.make_axes(ax2,location='bottom',fraction=0.1,shrink=0.95,aspect=8)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,norm=cNorm,**kw)
            cbar.set_label(r'projected density [$M_\odot/$pc$^2$]', labelpad=15, fontsize=0.75*fontsize)
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.tick_params(axis='x', which='major', pad=1, labelsize=0.5*fontsize)
            # plt.setp(cbar.ax.get_xticklabels())

        plt.setp(ax2.axes.yaxis,visible=False)
        plt.setp(ax2.axes.xaxis,visible=False)
        for k in ['top','bottom','left','right']:
            plt.setp(ax2.spines[k],visible=False)

        #first, x vs z in ax1
        circ = plt.Circle((0,0),radius_to_plot,facecolor='None',edgecolor=circcolor,lw=0.75)
        ax1.add_artist(circ)
        #second, x vs y in ax3
        circ = plt.Circle((0,0),radius_to_plot,facecolor='None',edgecolor=circcolor,lw=0.75)
        ax3.add_artist(circ)        
        #last, z vs y in ax4        
        circ = plt.Circle((0,0),radius_to_plot,facecolor='None',edgecolor=circcolor,lw=0.75)
        ax4.add_artist(circ)

    else:
        #edge on and face on 
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot([x1,x2],[y,y],lw=3,ls='-',color='w',label='_nolegend_')
        ax.text((x1+x2)/2.,y+width/50,scalelabel,color='w',ha='center',va='bottom')
       
        plt.setp(ax.get_yticklabels(),visible=False)
        plt.setp(ax.get_xticklabels(),visible=False)

        circ = plt.Circle((0,0),radius_to_plot,facecolor='None',edgecolor=circcolor,lw=0.75)
        ax.add_artist(circ)
    
    if rotate and return_rv:
        return fig,RV
    return fig








def viz_part_2d(part, center_position, outbase=None, outext='png',
           species=['star', 'gas', 'dark'],  part_indices={},
           kind='3panel', bsize=20, binsize=0.25, #method='hist', 
           plot_kwargs={
            'star':{'cmap':'magma', 'vmin':1e-2, 'vmax':10, }, 
            'dark':{'cmap':'cubehelix',}, 
            'gas': {'cmap':'viridis', }
            },
           nearby_method='scatter', nearby_kwargs={'s':1, 'marker':',' , 'edgecolor':'None','facecolor':'cyan','alpha':0.5},
           label=None,
           rotate=False, rotate_cut=-1, rotate_spec='star', rot_vec=None, return_rv=False, 
           circcolor='w', circradius=-1,
           cbar=True,verbose=False,figsize=None,fontsize=28,py2=False):
    '''
    visualize particles belonging to a halo/galaxy and nearby particles not belonging 
    to that halo/galaxy.  either can be plotted as either a 2D histogram or a set of 
    scatterpoints; the _kwargs are passed to the appropriate plotting command.

    Parameters
    ----------

    part : particle dictionary : particle data with particle species to visualize
    center_position: array-like: length 3 array giving the center
    outbase : string : start of a string to save the figures.  adds _{}.{}'.format(species_name, outext)
    outext : string : type of file to save
    species_name : list or string : partticle species to visualize
    part_indices : dictionary : a list for each species of indices to visualize.  
        nearby particles not in that list will be shown via nearby_method.
        don't need to provide a list for each pytpe; in that case, all nearby 
        particles are considered "mine" and there is no nearby_method done 
    kind :string : 
        one of ['3panel', 'faceon', 'edgeon'].  3 panel displays all 3 projections in a
        2x2 grid.  faceon and edgeon show just one view; if rotate is true, then they show
        that visualization; if false, you'll just get two different (but random) projections
    bsize : float : half size of the box to visualize (comoving kpc)
    binsize : float : size of the bins to use
    plot_kwargs : dictionary : kwargs passed to either ax.pcolormesh to make the plot
    nearby_method : string : 
        either 'hist' or 'scatter' to do either 2d histograms or scatter plots
        of partciles in the box but not associated with the targeted object.  if hist,
        then normalization is taken from plot_kwargs
    nearby_kwargs : dictionary : kwargs passed to either ax.scatter or ax.pcolormesh
    label : string : text added to the empty area in the 2x2 plot.  not shown if 
        face-on or edge-on image
    rotate : boolean : whether to rotate the images to align with principal axes
    rotate_cut : float : radius from the center to use to calculate rotation vectors
    rot_vec : array : rotation vectors to use; otherwise, calculated via rotate_cut
    return_rv : boolean : whether to return the calculated rotation vectors
    circcolor : string : color of the circle to put around the halo
    circradius : float : radius of the circle to plot around the halo, or <= 0 for None
    cbar : boolean : whether or not ot do a colorbar in the empty axes, or alongsize
    verbose : boolean : whether or not to print extra info
    '''

    assert kind in ['3panel','edgeon','faceon']

    import numpy as np    
    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec

    import utilities as ut
    from .low_level_utils import boxmsk, 

    try:
        if nearby_method.startswith('hist'):        nearby_method = 'hist'
        elif nearby_method.startswith('scatter'):   nearby_method = 'scatter'
    except AttributeError:
        pass
    nbins = int(np.ceil(bsize/binsize))

    if type(species) != list:
        species = [species]

    if rotate and rot_vec is None:
        if rotate_spec not in part:
            raise KeyError("Asked to rotate on {}, but that's not in the particle dictionary".format(rotate_spec))
        if rotate_spec not in species:
            print("Adding {} to species to viz because you're asking to rotate on it, so deal with it.".format(rotate_spec))
        elif species[0] != rotate_spec:
            species.pop(rotate_spec)
            species = [rotate_spec] + species  #do roate_spec first

    figures = {}
    for species_name in species:
        if rotate:
            if rot_vec is None:
                assert rotate_cut >= 0 and rotate_spec in part
                RV,EV,AR = ut.particle.get_principal_axes(part, species=rotate_spec, center_position=center_position, distance_max=rotate_cut)
            else:
                RV = rot_vec
            rot_pos = ut.particle.get_distances_wrt_center(part,species=species_name,center_position=center_position,distance_kind='vector',rotation=RV)
            particles_in_box = boxmsk(rot_pos, [0,0,0], bsize) #already shifted
        else:
            rot_pos = part[species_name]['position'] - center_position
            particles_in_box = boxmsk(part[species_name]['position'],center_position, bsize)   

        nbox = np.count_nonzero(particles_in_box)
        if species_name in part_indices:
            pind = part_indices[species_name]
            my_msk = np.empty_like(particles_in_box)
            my_msk[:] = False
            my_msk[pind] = True
        
            #particles assigned to this halo that are in the box:
            selected_msk = my_msk & particles_in_box
            nmine = np.count_nonzero(selected_msk)
        
            #particles that are assigned to this halo that are outside the box:
            missed_msk = my_msk & np.logical_not(particles_in_box)    
            nmissed = np.count_nonzero(missed_msk)
            
            #particles not assigned to this halo that are inside the box:
            unselected_msk = np.logical_not(my_msk) & particles_in_box
            nother = np.count_nonzero(unselected_msk)
        else:
            selected_msk = particles_in_box
            pind = None

        if verbose:
            print("Plotting {} particles in a cube of half side length = {} kpc".format(nbox, bsize))
            if pind is not None:
                print("{} of those ({:.1f}%) are associated with the halo"
                      .format(nmine,nmine*100./nbox))
                print("{} of the particles associated with the halo ({:.1f}%) are outside the box"
                      .format(nmissed,nmissed*100./part_indices.shape[0]))          
            
        my_pos = rot_pos[selected_msk]
        my_weights = part[species_name]['mass'][selected_msk]
            
        lim = [int(round(-bsize)),int(round(bsize))]
        tickloc = []
        tickstr = []

        if nbox == 0:
            print("No {} particles in the box; skipping them".format(species_name))
            continue

        this_kwargs = plot_kwargs[species_name]
        if 'cmap' in this_kwargs:
            cmap = matplotlib.cm.get_cmap(this_kwargs['cmap'])
            bgcolor = list(cmap(0)[:-1])
            scalecolor = 'w'
        elif nearby_method == 'hist' and 'cmap' in nearby_kwargs:
            cmap = matplotlib.cm.get_cmap(nearby_kwargs['cmap'])
            bgcolor = list(cmap(0)[:-1])
            scalecolor = 'w'
        else:
            cbar = False #can't do a colorbar without a colormap
            bgcolor = 'w'
            scalecolor = 'k'

        cvmin = this_kwargs.pop('vmin', None)
        cvmax = this_kwargs.pop('vmax', None)
        cNorm = LogNorm(vmin=cvmin,vmax=cvmax)

        if figsize is None:
            figsize = (max(mpl.rcParams['figure.figsize']), max(mpl.rcParams['figure.figsize']))

        fig = plt.figure(figsize=figsize)
        if kind == '3panel':
            ax1 = fig.add_subplot(221,facecolor=bgcolor,aspect='equal')   #x vs z
            ax2 = fig.add_subplot(222,facecolor='w')   #nothing; maybe a label
            ax3 = fig.add_subplot(223,facecolor=bgcolor,sharex=ax1,aspect='equal')   #x vs y -- shares the xaxis with the plot above it
            ax4 = fig.add_subplot(224,facecolor=bgcolor,sharey=ax3,aspect='equal')   #z vs y -- shares the y axis with the plot beside it

            ax3.set_xlabel(r'$x$', fontsize=fontsize)
            ax3.set_ylabel(r'$y$', fontsize=fontsize)
            ax4.set_xlabel(r'$z$', fontsize=fontsize)
            ax1.set_ylabel(r'$z$', fontsize=fontsize)
        elif cbar:
            gs = GridSpec(1,2,width_ratios=[12,1],wspace=0.02)
            ax = plt.subplot(gs[0],facecolor=bgcolor,aspect='equal')
            cax = plt.subplot(gs[1])
        else:
            ax = fig.add_subplot(111, facecolor=bgcolor, aspect='equal')

        xp,yp,zp = my_pos.T
        if xp.size:
            if kind == '3panel':
                #first, x vs z in ax1
                hist,xe,ye = np.histogram2d(xp,zp,bins=nbins,weights=my_weights)
                hist = convert_to_per_pc2(hist,xe[1]-xe[0])
                Y_eo,X_eo = np.meshgrid(xe,ye)
                im = ax1.pcolormesh(X_eo, Y_eo, hist, norm=cNorm, **this_kwargs) 

                #second, x vs y in ax3
                hist,xe,ye = np.histogram2d(xp,yp,bins=nbins,weights=my_weights)
                hist = convert_to_per_pc2(hist,xe[1]-xe[0])
                Y_eo,X_eo = np.meshgrid(xe,ye)
                im = ax3.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**this_kwargs) 

                #last, z vs y in ax4
                hist,xe,ye = np.histogram2d(zp,yp,bins=nbins,weights=my_weights)
                hist = convert_to_per_pc2(hist,xe[1]-xe[0])
                Y_eo,X_eo = np.meshgrid(xe,ye)
                im = ax4.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**this_kwargs) 
            else:
                if kind == 'edgeon':
                    #want x vs z:
                    xi = xp
                    yi = zp    
                elif kind == 'faceon':
                    #x vs y
                    xi = xp
                    yi = yp
                hist,xe,ye = np.histogram2d(xi,yi,bins=nbins,weights=my_weights)
                hist = convert_to_per_pc2(hist,xe[1]-xe[0])
                Y_eo,X_eo = np.meshgrid(xe,ye)
                im = ax.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**ided_kwargs)
                if cbar:
                    cbar = plt.colorbar(im,cax=cax)
                    cbar.set_label(r'projected density [$M_\odot$/pc$^2$]', labelpad=15)

        if pind is not None:
            other_pos = rot_pos[unselected_msk]
            other_weights = part[species_name]['mass'][unselected_msk]   
            xp,yp,zp = other_pos.T
            if xp.size:
                if nearby_method == 'hist':  
                    if kind == '3panel': 
                        #first, x vs z in ax1
                        hist,xe,ye = np.histogram2d(xp,zp,bins=nbins,weights=other_weights)
                        hist = convert_to_per_pc2(hist,xe[1]-xe[0])
                        Y_eo,X_eo = np.meshgrid(xe,ye)
                        im = ax1.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**nearby_kwargs) 

                        #second, x vs y in ax3
                        hist,xe,ye = np.histogram2d(xp,yp,bins=nbins,weights=other_weights)
                        hist = convert_to_per_pc2(hist,xe[1]-xe[0])
                        Y_eo,X_eo = np.meshgrid(xe,ye)
                        im = ax3.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**nearby_kwargs) 

                        #last, z vs y in ax4
                        hist,xe,ye = np.histogram2d(zp,yp,bins=nbins,weights=other_weights)
                        hist = convert_to_per_pc2(hist,xe[1]-xe[0])
                        Y_eo,X_eo = np.meshgrid(xe,ye)
                        im = ax4.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**nearby_kwargs)
                    else:
                        if kind == 'edgeon':
                            #want x vs z:
                            xi = xp
                            yi = zp    
                        elif kind == 'faceon':
                            #x vs y
                            xi = xp
                            yi = yp
                        hist,xe,ye = np.histogram2d(xi,yi,bins=nbins,weights=my_weights)
                        hist = convert_to_per_pc2(hist,xe[1]-xe[0])
                        Y_eo,X_eo = np.meshgrid(xe,ye)
                        im = ax.pcolormesh(X_eo,Y_eo,hist,norm=cNorm,**nearby_kwargs)
                
                elif nearby_method == 'scatter':   
                    if kind == '3panel':
                        #first, x vs z in ax1
                        ax1.scatter(xp,zp,**nearby_kwargs)

                        #second, x vs y in ax3
                        ax3.scatter(xp,yp,**nearby_kwargs)

                        #last, z vs y in ax4
                        ax4.scatter(zp,yp,**nearby_kwargs)
                    else:
                        if kind == 'edgeon':
                            #want x vs z:
                            xi = xp
                            yi = zp    
                        elif kind == 'faceon':
                            #x vs y
                            xi = xp
                            yi = yp
                        ax.scatter(xi,yi,**nearby_kwargs)  
                elif nearby_method in ['none','None',None]:
                    pass
                else:
                    print("!! don't know method for plotting particles as {}; not plotting nearby particles".format(other_method))


        width = 2*bsize
        if width > 1000:
            scalesize = 300
            scalelabel = r'$\mathrm{300\,kpc}$'
        elif width < 1000 and width > 500:
            scalesize = 100
            scalelabel = r'$\mathrm{100\,kpc}$'
        elif width < 600 and width > 100:
            scalesize = 50
            scalelabel = r'$\mathrm{50\,kpc}$'
        elif width > 50 and width < 100:
            scalesize = 10
            scalelabel = r'$\mathrm{10\,kpc}$'
        elif width <= 50 and width > 9:
            scalesize = 5
            scalelabel = r'$\mathrm{5\,kpc}$'
        elif width < 10 and width > 2:
            scalesize = 1
            scalelabel = r'$\mathrm{1\,kpc}$'
        else:
            scalesize = 0.1
            scalelabel = r'$\mathrm{100\,pc}$'

        x1 = - bsize + width/20
        x2 = x1+scalesize
        y = width/20 - bsize

        if kind == '3panel':
            ax1.set_xlim(lim)
            ax1.set_ylim(lim)

            ax2.set_xlim(lim)
            ax2.set_ylim(lim)

            ax3.set_xlim(lim)
            ax3.set_ylim(lim)

            ax4.set_xlim(lim)
            ax4.set_ylim(lim)

            ax1.set_xticks([])
            ax1.set_yticks([])
            plt.setp(ax1.get_yticklabels(),visible=False)
            plt.setp(ax1.get_xticklabels(),visible=False)

            ax2.set_xticks([])
            ax2.set_yticks([])
            plt.setp(ax2.get_yticklabels(),visible=False)
            plt.setp(ax2.get_xticklabels(),visible=False)

            ax3.set_xticks([])
            ax3.set_yticks([])
            plt.setp(ax3.get_yticklabels(),visible=False)
            plt.setp(ax3.get_xticklabels(),visible=False)

            ax4.set_xticks([])
            ax4.set_yticks([])
            plt.setp(ax4.get_yticklabels(),visible=False)
            plt.setp(ax4.get_xticklabels(),visible=False)

            fig.subplots_adjust(left=0.05,bottom=0.05,top=0.95,right=0.95,hspace=0.025,wspace=0.025)

            ax1.plot([x1,x2],[y,y],lw=3,ls='-',color=scalecolor,label='_nolegend_')
            ax1.text((x1+x2)/2.,y+width/50,scalelabel,color=scalecolor,ha='center',va='bottom',fontsize=0.75*fontsize)

            ax3.plot([x1,x2],[y,y],lw=3,ls='-',color=scalecolor,label='_nolegend_')
            ax3.text((x1+x2)/2.,y+width/50,scalelabel,color=scalecolor,ha='center',va='bottom',fontsize=0.75*fontsize)

            ax4.plot([x1,x2],[y,y],lw=3,ls='-',color=scalecolor,label='_nolegend_')
            ax4.text((x1+x2)/2.,y+width/50,scalelabel,color=scalecolor,ha='center',va='bottom',fontsize=0.75*fontsize)

            if label is not None:
                ax2.text(0.5,0.1,label,ha='center',va='bottom',transform=ax2.transAxes,color='k',fontsize=fontsize)

            if cbar:
                cax,kw = mpl.colorbar.make_axes(ax2,location='bottom',fraction=0.1,shrink=0.95,aspect=8)
                cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,norm=cNorm,**kw)
                cbar.set_label(r'projected density [$M_\odot/$pc$^2$]', labelpad=15, fontsize=0.75*fontsize)
                cbar.ax.xaxis.set_ticks_position('top')
                cbar.ax.xaxis.set_label_position('top')
                cbar.ax.tick_params(axis='x', which='major', pad=1, labelsize=0.5*fontsize)
                # plt.setp(cbar.ax.get_xticklabels())        

            plt.setp(ax2.axes.yaxis,visible=False)
            plt.setp(ax2.axes.xaxis,visible=False)
            for k in ['top','bottom','left','right']:
                plt.setp(ax2.spines[k],visible=False)

            if circradius > 0:
                #first, x vs z in ax1
                circ = plt.Circle((0,0),circradius,facecolor='None',edgecolor=circcolor,lw=0.75)
                ax1.add_artist(circ)
                #second, x vs y in ax3
                circ = plt.Circle((0,0),circradius,facecolor='None',edgecolor=circcolor,lw=0.75)
                ax3.add_artist(circ)        
                #last, z vs y in ax4        
                circ = plt.Circle((0,0),circradius,facecolor='None',edgecolor=circcolor,lw=0.75)
                ax4.add_artist(circ)

        else:
            #edge on and face on 
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.plot([x1,x2],[y,y],lw=3,ls='-',color='w',label='_nolegend_')
            ax.text((x1+x2)/2.,y+width/50,scalelabel,color='w',ha='center',va='bottom')
           
            plt.setp(ax.get_yticklabels(),visible=False)
            plt.setp(ax.get_xticklabels(),visible=False)

            if circradius > 0:
                circ = plt.Circle((0,0),radius_to_plot,facecolor='None',edgecolor=circcolor,lw=0.75)
                ax.add_artist(circ)

        if outbase is not None:
            outname = outbase+'_{}.{}'.format(species_name, outext)
            plt.savefig(outname)
        figures[species_name] = fig

    if rotate and return_rv:
        return figures,RV
    return figures






def viz_with_halos(part,hal=None,species_name='star',kind='3panel',
                   center_position=np.r_[0,0,0],bsize=300,nbins=1e3,
                   figsize=(13,13),label=None,
                   cvmin=1e-5,cvmax=1,cmap='cubehelix',
                   rotate=False,rotate_cut=0.5,rot_vec=None,return_rv=False,
                   circcolor='w',hal_indices=[],halcut={'star.mass':[3e4,np.inf]},
                   circprop='star.radius.90',circmult=1.,
                   cbar=True,verbose=False,
                   highlight_hali=None, plot_others=True,onebyone='',onebyone_highlight='cyan',
                   ):   
    '''
    visualize particles belonging to a halo/galaxy and nearby particles not belonging 
    to that halo/galaxy.  either can be plotted as either a 2D histogram or a set of 
    scatterpoints; the _kwargs are passed to the appropriate plotting command.

    Parameters
    ----------

    part : particle dictionary : particle data with particle species to visualize
    hal : halo catalog : 
        halo catalog with positions to circle on the visualizations, or None to skip it
    species_name : string : particle species to visualize
    kind :string : 
        one of ['3panel', 'faceon', 'edgeon'].  3 panel displays all 3 projections in a
        2x2 grid.  faceon and edgeon show just one view; if rotate is true, then they show
        that visualization; if false, you'll just get two different (but random) projections
    center_position : array : x,y,z position to center on
    bsize : float : half size of the box to visualize (comoving kpc)
    nbins : int : number of bins to use for 2dhistograms
    cvmin : float : minimum of the color normalization for histograms
    cvmax : float : max of the color normalizations
    cmap : string : colormap to use
    figsize : tuple : figure size for matplotlib
    fontsize : float : fontsize for matplot
    label : string : text added to the empty area in the 2x2 plot.  not shown if 
        face-on or edge-on image
    rotate : boolean : whether to rotate the images to align with principal axes
    rotate_cut : float : fraction of the box size to use to calculate rotation vectors
    rot_vec : array : rotation vectors to use
    return_rv : boolean : whether to return the calculated rotation vectors
    circcolor : string : color of the circles to put around the halos/galaxies
    hal_indices : list : list of indices of halos to circle; ignores halcut if provided
    halcut : dictionary : dictionary to use to select the halo, composed of a 
        property and a dictionary giving the min/max allowed for that property
    circprop : string : what property to use for the circles radii
    circmult : float : what to multiple the circle radius by
    cbar : boolean : whether or not ot do a colorbar in the empty axes, or along side
    verbose : boolean : whether or not to print extra info
    onebyone : string : 
        pass a non-empty string to create a visualization for each hal_indices (or 
        each halo that passes halcut); string is taken as the base of the filename 
    onebyone_highlight : string : 
        pass a non-empty string as a color to use for the halos as you highlight them,
        or pass in an empty string to make a separate image for each halo (w/o the other
        halos circled at all) using circcolor
    '''
    assert kind in ['3panel','edgeon','faceon']

    import numpy as np    
    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec

    import utilities as ut
    import numpy as np

    from .low_level_utils import boxmsk

    if rotate:
        if rot_vec is None:
            RV,EV,AR = ut.particle.get_principal_axes(part,species=species_name,center_position=center_position,distance_max=bsize*rotate_cut)
        else:
            RV = rot_vec
        # rot_pos = ut.particle.get_distances_wrt_center(part,species=species_name,center_position=center_position,distance_kind='vector',rotation=RV)
        rot_pos = ut.particle.get_distances_wrt_center(part,species=species_name,center_position=center_position,rotation=RV)
        msk = boxmsk(rot_pos,[0,0,0],bsize) #already shifted
        xp,yp,zp = rot_pos[msk].T  #already shifted
    else:
        msk = boxmsk(part[species_name]['position'],center_position,bsize)   
        xp,yp,zp = (part[species_name]['position'][msk] - center_position).T   
    weights = part[species_name]['mass'][msk]

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from math import ceil,floor
    import matplotlib
    import utilities as ut

    lim = [int(round(-bsize)),int(round(bsize))]
#     tickloc = [int(round(-bsize*0.5)),0,int(round(bsize*0.5))]
#     tickstr = ['$'+str(kk)+'$' for kk in tickloc]
    tickloc = []
    tickstr = []

    cmap = matplotlib.cm.get_cmap(cmap)
    bgcolor = list(cmap(0)[:-1])
    cNorm = LogNorm(vmin=cvmin,vmax=cvmax)

    fig = plt.figure(figsize=figsize)
    if kind == '3panel':
        ax1 = fig.add_subplot(221,facecolor=bgcolor,aspect='equal')   #x vs z
        ax2 = fig.add_subplot(222,facecolor='w')   #nothing; maybe a label
        ax3 = fig.add_subplot(223,facecolor=bgcolor,sharex=ax1,aspect='equal')   #x vs y -- shares the xaxis with the plot above it
        ax4 = fig.add_subplot(224,facecolor=bgcolor,sharey=ax3,aspect='equal')   #z vs y -- shares the y axis with the plot beside it

        ax3.set_xlabel(r'$x$')
        ax3.set_ylabel(r'$y$')
        ax4.set_xlabel(r'$z$')
        ax1.set_ylabel(r'$z$')
    else:
        gs = GridSpec(1,2,width_ratios=[12,1],wspace=0.02)
        ax = plt.subplot(gs[0],facecolor=bgcolor,aspect='equal')
        cax = plt.subplot(gs[1])

    if kind == '3panel':
        #first, x vs z in ax1
        hist,xe,ye = np.histogram2d(xp,zp,bins=nbins,weights=weights)
        hist = convert_to_per_pc2(hist,xe[1]-xe[0])
        Y_eo,X_eo = np.meshgrid(xe,ye)
        im = ax1.pcolormesh(X_eo,Y_eo,hist,norm=LogNorm(),cmap=cmap,vmax=cvmax,vmin=cvmin) 

        #second, x vs y in ax3
        hist,xe,ye = np.histogram2d(xp,yp,bins=nbins,weights=weights)
        hist = convert_to_per_pc2(hist,xe[1]-xe[0])
        Y_eo,X_eo = np.meshgrid(xe,ye)
        im = ax3.pcolormesh(X_eo,Y_eo,hist,norm=LogNorm(),cmap=cmap,vmax=cvmax,vmin=cvmin) 

        #last, z vs y in ax4
        hist,xe,ye = np.histogram2d(zp,yp,bins=nbins,weights=weights)
        hist = convert_to_per_pc2(hist,xe[1]-xe[0])
        Y_eo,X_eo = np.meshgrid(xe,ye)
        im = ax4.pcolormesh(X_eo,Y_eo,hist,norm=LogNorm(),cmap=cmap,vmax=cvmax,vmin=cvmin) 
    else:
        if kind == 'edgeon':
            #want x vs z:
            xi = xp
            yi = zp    
        elif kind == 'faceon':
            #x vs y
            xi = xp
            yi = yp
        hist,xe,ye = np.histogram2d(xi,yi,bins=nbins,weights=weights)
        hist = convert_to_per_pc2(hist,xe[1]-xe[0])
        Y_eo,X_eo = np.meshgrid(xe,ye)
        im = ax.pcolormesh(X_eo,Y_eo,hist,norm=LogNorm(),cmap=cmap,vmax=cvmax,vmin=cvmin) 
        if cbar:
            cbar = plt.colorbar(im,cax=cax)
            cbar.set_label(r'projected density [$M_\odot$/pc$^2$]', labelpad=15)
    
    width = 2*bsize
    if width > 1000:
        scalesize = 300
        scalelabel = r'$\mathrm{300\,kpc}$'
    elif width < 1000 and width > 500:
        scalesize = 100
        scalelabel = r'$\mathrm{100\,kpc}$'
    elif width < 600 and width > 100:
        scalesize = 50
        scalelabel = r'$\mathrm{50\,kpc}$'
    elif width > 50 and width < 100:
        scalesize = 10
        scalelabel = r'$\mathrm{10\,kpc}$'
    elif width <= 50 and width > 9:
        scalesize = 5
        scalelabel = r'$\mathrm{5\,kpc}$'
    elif width < 10 and width > 2:
        scalesize = 1
        scalelabel = r'$\mathrm{1\,kpc}$'
    else:
        scalesize = 0.1
        scalelabel = r'$\mathrm{100\,pc}$'

    x1 = - bsize + width/20
    x2 = x1+scalesize
    y = width/20 - bsize

    if kind == '3panel':
        ax1.set_xlim(lim)
        ax1.set_ylim(lim)

        ax2.set_xlim(lim)
        ax2.set_ylim(lim)

        ax3.set_xlim(lim)
        ax3.set_ylim(lim)

        ax4.set_xlim(lim)
        ax4.set_ylim(lim)

        ax1.set_xticks([])
        ax1.set_yticks([])
        plt.setp(ax1.get_yticklabels(),visible=False)
        plt.setp(ax1.get_xticklabels(),visible=False)

        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.setp(ax2.get_yticklabels(),visible=False)
        plt.setp(ax2.get_xticklabels(),visible=False)

        ax3.set_xticks([])
        ax3.set_yticks([])
        plt.setp(ax3.get_yticklabels(),visible=False)
        plt.setp(ax3.get_xticklabels(),visible=False)

        ax4.set_xticks([])
        ax4.set_yticks([])
        plt.setp(ax4.get_yticklabels(),visible=False)
        plt.setp(ax4.get_xticklabels(),visible=False)

        fig.subplots_adjust(left=0.05,bottom=0.05,top=0.95,right=0.95,hspace=0.075,wspace=0.075)

        ax1.plot([x1,x2],[y,y],lw=3,ls='-',color='w',label='_nolegend_')
        ax1.text((x1+x2)/2.,y+width/50,scalelabel,color='w',ha='center',va='bottom')

        ax3.plot([x1,x2],[y,y],lw=3,ls='-',color='w',label='_nolegend_')
        ax3.text((x1+x2)/2.,y+width/50,scalelabel,color='w',ha='center',va='bottom')

        ax4.plot([x1,x2],[y,y],lw=3,ls='-',color='w',label='_nolegend_')
        ax4.text((x1+x2)/2.,y+width/50,scalelabel,color='w',ha='center',va='bottom')

        if label is not None:
            ax2.text(0.5,0.1,label,ha='center',va='bottom',transform=ax2.transAxes,color='k',fontsize='small')

        if cbar:
            cax,kw = mpl.colorbar.make_axes(ax2,location='bottom',fraction=0.1,shrink=0.95,aspect=8)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap,norm=cNorm,**kw)
            cbar.set_label(r'projected density ($M_\odot/$pc$^2$', labelpad=15)
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.tick_params(axis='x', which='major', pad=1)
            # plt.setp(cbar.ax.get_xticklabels(),fontsize=fontsize*0.5)      

        plt.setp(ax2.axes.yaxis,visible=False)
        plt.setp(ax2.axes.xaxis,visible=False)
        for k in ['top','bottom','left','right']:
            plt.setp(ax2.spines[k],visible=False)
    else:
        #edge on and face on 
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot([x1,x2],[y,y],lw=3,ls='-',color='w',label='_nolegend_')
        ax.text((x1+x2)/2.,y+width/50,scalelabel,color='w',ha='center',va='bottom')
       
        plt.setp(ax.get_yticklabels(),visible=False)
        plt.setp(ax.get_xticklabels(),visible=False)


    if hal is not None:
        if rotate:
            mycen = ut.coordinate.get_coordinates_rotated(hal['position']-center_position,rotation_vectors=RV)
        else:
            mycen = hal['position'] - center_position

        if not len(hal_indices):
            halmsk = boxmsk(mycen,[0,0,0],bsize)
            if verbose:
                print("{} total halos in the box".format(np.count_nonzero(halmsk)))
            
            prop_msk = np.empty_like(halmsk)
            prop_msk[:] = False
            hal_indices = ut.catalog.get_indices_catalog(hal,halcut)
            prop_msk[hal_indices] = True
            my_msk = halmsk & prop_msk
            if verbose:
                print("{} halos in the box passed the cut".format(np.count_nonzero(my_msk)))
        else:
            my_msk = np.empty(hal['position'].shape[0],dtype=bool)*False
            my_msk[hal_indices] = True
        if plot_others and ((len(onebyone_highlight)) or (not len(onebyone))):  #either not doing onebyone at all, or want to do onebyone but highlighting the selected one (in which case I first plot all of the circles)
            print("Circling {} halos in the box".format(np.count_nonzero(my_msk)))
            for ii in range(my_msk.shape[0]):
                if my_msk[ii]:
                    cen = mycen[ii]
                    r = hal[circprop][ii]*circmult
                    if kind == '3panel':
                        #first, x vs z in ax1
                        circ1 = plt.Circle((cen[0],cen[2]),r,facecolor='None',edgecolor=circcolor,lw=0.75)
                        ax1.add_artist(circ1)
                        #second, x vs y in ax3
                        circ3 = plt.Circle((cen[0],cen[1]),r,facecolor='None',edgecolor=circcolor,lw=0.75)
                        ax3.add_artist(circ3)        
                        #last, z vs y in ax4        
                        circ4 = plt.Circle((cen[2],cen[1]),r,facecolor='None',edgecolor=circcolor,lw=0.75)
                        ax4.add_artist(circ4)
                    else:
                        if kind == 'edgeon':
                            #want x vs z:
                            xi = cen[0]
                            yi = cen[2]
                        elif kind == 'faceon':
                            #x vs y
                            xi = cen[0]
                            yi = cen[1]
                        circ = plt.Circle((xi,yi),r,facecolor='None',edgecolor=circcolor,lw=0.75)   
                        ax.add_artist(circ)  

        if highlight_hali is not None:
            cen = mycen[highlight_hali]
            r = hal.prop(circprop)[highlight_hali]*circmult
            if kind == '3panel':
                #first, x vs z in ax1
                circ1 = plt.Circle((cen[0],cen[2]),r,facecolor='None',edgecolor=onebyone_highlight,lw=0.75)
                ax1.add_artist(circ1)
                #second, x vs y in ax3
                circ3 = plt.Circle((cen[0],cen[1]),r,facecolor='None',edgecolor=onebyone_highlight,lw=0.75)
                ax3.add_artist(circ3)        
                #last, z vs y in ax4        
                circ4 = plt.Circle((cen[2],cen[1]),r,facecolor='None',edgecolor=onebyone_highlight,lw=0.75)
                ax4.add_artist(circ4)
            else:
                if kind == 'edgeon':
                    #want x vs z:
                    xi = cen[0]
                    yi = cen[2]
                elif kind == 'faceon':
                    #x vs y
                    xi = cen[0]
                    yi = cen[1]
                circ = plt.Circle((xi,yi),r,facecolor='None',edgecolor=circcolor,lw=0.75)   
                ax.add_artist(circ)            

        elif len(onebyone):
            if len(onebyone_highlight):
                color = onebyone_highlight
            else:
                color = circcolor

            for ii in range(my_msk.shape[0]):
                if my_msk[ii]:
                    cen = mycen[ii]
                    r = hal[circprop][ii]*circmult
                    if kind == '3panel':
                        #first, x vs z in ax1
                        circ1 = plt.Circle((cen[0],cen[2]),r,facecolor='None',edgecolor=color,lw=0.75)
                        ax1.add_artist(circ1)
                        #second, x vs y in ax3
                        circ3 = plt.Circle((cen[0],cen[1]),r,facecolor='None',edgecolor=color,lw=0.75)
                        ax3.add_artist(circ3)        
                        #last, z vs y in ax4        
                        circ4 = plt.Circle((cen[2],cen[1]),r,facecolor='None',edgecolor=color,lw=0.75)
                        ax4.add_artist(circ4)

                        artists = [circ1,circ3,circ4]
                    else:
                        if kind == 'edgeon':
                            #want x vs z:
                            xi = cen[0]
                            yi = cen[2]
                        elif kind == 'faceon':
                            #x vs y
                            xi = cen[0]
                            yi = cen[1]
                        circ = plt.Circle((xi,yi),r,facecolor='None',edgecolor=color,lw=0.75)   
                        ax.add_artist(circ)  
                        artists = [circ]

                    plt.savefig(onebyone+'_{}.png'.format(ii))
                    for artist in artists:
                        artist.remove()
                    del artists
        if return_rv:
            return fig,RV
        return fig