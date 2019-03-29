#!/usr/bin/env python

import numpy as np
from tqdm import tqdm
from astropy.constants import G as Ggrav

from .low_level_utils import fast_dist

G = Ggrav.to('kpc Msun**-1 km**2 s**-2').value

def all_profiles(bins, positions, velocities, masses, two_dimensional=False, zcut=None, 
    ages=None, pbar_msg='Making profiles"', nexpr=False):
    """
    assumes all positions and velocities are rotated in the same way, such 
        that the angular momentum axis aligns with the z axis

    if two_dimensional == False, then compute: 
        M(<r), M(r), rho = M(r)/dV, Vcirc = sqrt(GM(<r)/r), mag J(r), mag J(<r), J_z(r), J_z(<r)  
    if two_dimensional == True, then compute:
        M(<R), M(R), rho = M(R)/dA, Vcirc = mean(vx**2 + vy**2), mag J(R), mag J(<R), J_z(R), J_z(<R)
    
    :bins : array-like : sorted (from small to large) bin edges to use
    :positions : array-like :  particle positions, rotated such that z aligns with angular momentum axis
    :velocities : array-like : particle velocities, rotated in the same way as the positions
    :masses : array-like : particle masses, in the same order as positions and velocities
    :two_dimensional : bool : whether or not to do 2D profiles
    :pbar_msg: str : what to print for the pbar (total mass and number of particles is appended)
    :nexpr : bool : whether or not to try to use numexpr to try to speed up the calculation
    """
    if nexpr:
        from numexpr import evaluate
        print("Using numexpr for the masking and summing masses")


    #work from outside in, throwing away particles as I no longer need them
    assert positions.shape[0] == velocities.shape[0] == masses.shape[0]

    m_of_r = np.empty(bins.size)
    J_of_r = np.empty(bins.size)
    Jz_of_r = np.empty(bins.size)
    Jz_inside_r = np.empty(bins.size)
    JinsideR = np.empty(bins.size)

    specJinsideR = np.zeros(bins.size)
    specJ_of_r = np.zeros(bins.size)
    specJz_of_r = np.zeros(bins.size)    
    specJz_insideR = np.zeros(bins.size)

    if ages is not None:
        age_of_r = np.zeros(bins.size)

    density = np.empty_like(m_of_r)
    if two_dimensional:    
        vcirc = np.zeros(bins.size)

    if two_dimensional:
        x,y,z = positions.T
        distances = np.sqrt(x**2 + y**2)            #distances are in the plane of the galaxy
    else:
        distances = fast_dist(positions)            #center assumed to be at (0,0,0)

    #throw away any particles beyond my last bin edge
    msk = distances <= bins.max()
    if two_dimensional:
        msk = msk & (np.abs(z) <= zcut)

    positions = positions[msk]
    velocities = velocities[msk]
    masses = masses[msk]
    distances = distances[msk]
    if ages is not None:
        ages = ages[msk]
    if two_dimensional:
        x = x[msk]
        y = y[msk]

    #compute (angular) momenta for the particles: 
    pvec = (velocities.T*masses).T   #velocities should already have the halo at 
    Jvec = np.cross(positions,pvec)  #J = r cross p, and pos is assumed to have the halo at 0,0,0
    del pvec
    Jz = Jvec[:,2]

    if two_dimensional:            
        #calculate circular velocities:
        vx,vy = velocities[:,0],velocities[:,1]  #velocities in the plane of the disk
        V = np.vstack((vx,vy)).T                 #velocity vector in the plane of the disk
        R = np.vstack((x,y)).T                   #distance vector in the plane of the disk

        #use the definition of the dot product to find the angle between R and V, theta
        # a dot b == mag(a) * mag(b) * cos(theta) 
        # => cos(theta) == a dot b / (mag(a) * mag(b))
        R_dot_V = np.sum(R*V, axis=1)   #checked by hand -- does the dot product of R[ii] w/ V[ii]
        mag_V = np.linalg.norm(V, axis=1)
        mag_R = np.linalg.norm(R, axis=1)  #checked by hand -- gives the magnitdue of R[ii]
        if careful:
            assert (mag_R == distances).all()   #should be identically true
        theta = np.arccos( R_dot_V / (mag_R * mag_V) )

        #now that I know the angle, the circular velocity of each particle is going to be 
        #the magnitude of each velocity in the plane of the disk times the sin of angle between R and V 
        #-- if the angle is 0, then all the velocity is radial; if it's pi/2, then all the velocity is tangential (circular)
        circular_velocities = mag_V*np.sin(theta)
        
        #handle any nan (i.e. either R or V == 0) by replacing with a 0
        print("Replacing {} NaNs with 0".format(np.count_nonzero(np.isnan(circular_velocities))))
        circular_velocities[np.isnan(circular_velocities)] = 0

        #clean up to save memory
        del R,V,theta

    assert (np.sort(bins) == bins).all()        #make sure this is true because otherwise return will be nonsense since I use cumsum at the end
    rev_bins = bins[::-1]

    if two_dimensional:
        pbar_msg += '; Mtot(R < {:.0f} kpc, Z < {:.1f} kpc)'.format(bins.max(), zcut)
    else:
        pbar_msg += '; Mtot(r < {:.0f} kpc)'.format(bins.max())
    pbar_msg += ' = {:.2g} Msun, {:,} particles)'.format(np.sum(masses),masses.size)

    for ii in tqdm(range(len(rev_bins)), pbar_msg):
        rhigh = rev_bins[ii]
        if ii == len(rev_bins)-1:
            rlow = 0
        else:
            rlow = rev_bins[ii+1]
        assert rlow < rhigh   

        if two_dimensional:
            shell_vol = 4.*np.pi*(rhigh**2 - rlow**2)
        else:
            shell_vol = 4./3.*np.pi*(rhigh**3 - rlow**3)

        if nexpr:
            # within_rhigh = evaluate("(distances <= rhigh)")   #No need to do this -- I trim the particles before the loop and within the loop, so everything is within rhigh trivially
            minsider = evaluate("sum(masses)")
            inbin = evaluate("(distances > rlow)")
            thism = evaluate("sum(where(inbin,masses,0))")      #sum up the masses where inbin, 0 otherwise
            Jz_of_r[ii] = evaluate("sum(where(inbin,Jz,0))")
            Jz_inside_r[ii] = evaluate("sum(Jz)")
            #particles that are within rhigh but not in the bin.  equivalent to  (within_rhigh) & (logical_not( (distances>rlow) & (within_rhigh) )
            #equivalent to False if not within_rhigh, so throws away outer particles
            #equivalent to True & logical_not(True & True) = True & not(True) = True & False = False if distances > rlow and distances < rhigh
            #equivalent to True & not(False & True) = True & not(False) = True if distances <= rlow 
            # keep = evaluate("~inbin")     #but since I trim the particles so within_rhigh is trivially true (see above), this just reduces to not inbin, so no reason to calculate/store that     
        else:
            # within_rhigh = distances <= rhigh
            inbin = (distances > rlow) #&(within_rhigh)       #works for both 2D and 3D
            minsider = np.sum(masses)
            thism = np.sum(masses[inbin])
            # keep = within_rhigh & (~inbin)  #save logic as above
            Jz_of_r[ii] = np.sum(Jz[inbin])                 #just the z angular momentum for the particles int he bin, allowed to cancel
            Jz_inside_r[ii] = np.sum(Jz)      #Jz of all the particles inside R.  should be smoother.

        m_of_r[ii] = thism
        density[ii] = thism/shell_vol

        #norm of the vector sum (sum(Jx), sum(Jy), sum(Jz)) of the angular momentum in the bin -- no need to mass weight because J is mass weighted
        J_of_r[ii] = np.linalg.norm(np.sum(Jvec[inbin],axis=0))

        #Do the same for all the particles inside the max of this bin; different because these can cancel differently
        JinsideR[ii] = np.linalg.norm(np.sum(Jvec,axis=0))       #remember that everything is within the max of this bin

        #normalize all those to the approrpiate specific value if m > 0.  
        if thism > 0:
            specJ_of_r[ii] = J_of_r[ii]/thism
            specJz_of_r[ii] = Jz_of_r[ii]/thism
            if two_dimensional:  vcirc[ii] = np.average(circular_velocities[inbin],weights=masses[inbin])
            if ages is not None:    age_of_r[ii] = np.average(ages[inbin],weights=masses[inbin])
        if minsider > 0:
            specJinsideR[ii] = JinsideR[ii]/minsider
            specJz_insideR[ii] = Jz_inside_r[ii]/minsider

        distances = distances[~inbin]
        masses = masses[~inbin]
        positions = positions[~inbin]
        velocities = velocities[~inbin]
        Jvec = Jvec[~inbin]
        Jz = Jz[~inbin]
        if two_dimensional:    circular_velocities = circular_velocities[~inbin]
        if ages is not None:    ages = ages[~inbin]

    #swap everything back around so that I go from the inside out so that I can cumsum.  remember bins is already sorted because I didn't swap it; I created rev_bins.
    density = density[::-1]
    m_of_r = m_of_r[::-1]

    J_of_r = J_of_r[::-1]
    Jz_of_r = Jz_of_r[::-1]
    JinsideR = JinsideR[::-1]
    Jz_inside_r = Jz_inside_r[::-1]

    specJ_of_r = specJ_of_r[::-1]
    specJz_of_r = specJz_of_r[::-1]
    specJinsideR = specJinsideR[::-1]
    specJz_insideR = specJz_insideR[::-1]
    if ages is not None:
        age_of_r = age_of_r[::-1]

    mltr = np.cumsum(m_of_r)
    Jltr = np.cumsum(J_of_r)
    Jzltr = np.cumsum(Jz_of_r)
    specJltr = np.cumsum(specJ_of_r)
    specJzltr = np.cumsum(specJz_of_r)

    #don't cumsum the "inside R" lines -- doesn't make much sense

    if two_dimensional==False:
        #calculate keplerian circular velocity 
        vcirc = np.sqrt(G*mltr/bins)        #remember that bins didn't get reversed
    else:
        vcirc = vcirc[::-1]

    #remember this gets saved directly, so be good about naming!
    end = 'R' if two_dimensional else 'r'
    toreturn = {
            'density'             : density,
            'M.of.'+end           : m_of_r,
            'J.of.'+end           : J_of_r,
            'Jz.of.'+end          : Jz_of_r,
            'J.inside'+end        : JinsideR,
            'Jz.inside'+end       : Jz_inside_r,
            'spec.J.of.'+end      : specJ_of_r,
            'spec.Jz.of.'+end     : specJz_of_r,
            'spec.Jinside'+end    : specJinsideR,
            'spec.Jz.insideR'+end : specJz_insideR,
            'M.lt.'+end           : mltr,
            'J.lt.'+end           : Jltr,
            'Jz.lt.'+end          : Jzltr,
            'spec.J.lt.'+end      : specJltr,
            'spec.Jz.lt.'+end     : specJzltr,
            'vcirc'               : vcirc,
                }
    if ages is not None:
        toreturn['age.of.'+end] = age_of_r

    return toreturn


def particle_mass_profiles(part, species='all', bins=None, center_position=None, **kwargs):
    '''
    given part (a particle dictionary), call mass_profiles on the particles

    bins can be either:
        * None -- defaults to logspace(-2, 0.5, 150)
        * raw bin edges -- passed directly
        * single integer -- defaults to logspace(-2, 0.5, bins)

    '''
    import utilities as ut

    species = ut.particle.parse_species(part, species)
    center_position = ut.particle.parse_property(part, 'center_position', center_position)

    npart = np.sum([part[spec]['mass'].size for spec in species])

    positions = np.empty((npart, 3))
    masses = np.empty(npart)

    left = 0
    for spec in species:
        right = left + part[spec]['mass'].size

        positions[left:right] = part[spec]['position']
        masses[left:right] = part[spec]['mass']

        left = right

    # shift so that the center is at [0, 0, 0]:
    positions -= center_position

    # now handle the bins:
    if bins is None:
        bins = np.logspace(-2, 0.5, 150)
    elif isinstance(bins, int):
        bins = np.logspace(-2, 0.5, bins)
    elif len(bins) == 3:
        bins = np.logspace(bins[0], bins[1], bins[2])

    assert not np.isscalar(bins)

    return mass_profiles(bins, positions, masses, **kwargs)

def mass_profiles(bins, positions, masses, pbar_msg='Making mass profiles', nexpr=False):
    """
    computes: 
        M(<r), M(r), rho = M(r)/dV, Vcirc = sqrt(GM(<r)/r)
    
    :bins : array-like : sorted (from small to large) bin edges to use
    :positions : array-like :  particle positions, with the center at 0,0,0
    :masses : array-like : particle masses, in the same order as positions and velocities
    :pbar_msg: str : what to print for the pbar (total mass and number of particles is appended)
    :nexpr : bool : whether or not to try to use numexpr to try to speed up the calculation
    """
    if nexpr:
        from numexpr import evaluate
        print("Using numexpr for the masking and summing masses")


    #work from outside in, throwing away particles as I no longer need them
    assert positions.shape[0] == masses.shape[0]

    m_of_r = np.empty(bins.size)
    density = np.empty_like(m_of_r)
    distances = fast_dist(positions)            #center assumed to be at (0,0,0)

    #throw away any particles beyond my last bin edge
    msk = distances <= bins.max()
    positions = positions[msk]
    masses = masses[msk]
    distances = distances[msk]

    assert (np.sort(bins) == bins).all()        #make sure this is true because otherwise return will be nonsense since I use cumsum at the end
    rev_bins = bins[::-1]

    pbar_msg += '; Mtot(r < {:.0f} kpc)'.format(bins.max())
    pbar_msg += ' = {:.2g} Msun, {:,} particles)'.format(np.sum(masses),masses.size)

    for ii in tqdm(range(len(rev_bins)), pbar_msg):
        rhigh = rev_bins[ii]
        if ii == len(rev_bins)-1:
            rlow = 0
        else:
            rlow = rev_bins[ii+1]
        assert rlow <= rhigh   

        shell_vol = 4./3.*np.pi*(rhigh**3 - rlow**3)
        if nexpr:
            # within_rhigh = evaluate("(distances <= rhigh)")   #No need to do this -- I trim the particles before the loop and within the loop, so everything is within rhigh trivially
            minsider = evaluate("sum(masses)")
            inbin = evaluate("(distances > rlow)")
            thism = evaluate("sum(where(inbin,masses,0))")      #sum up the masses where inbin, 0 otherwise
            #particles that are within rhigh but not in the bin.  equivalent to  (within_rhigh) & (logical_not( (distances>rlow) & (within_rhigh) )
            #equivalent to False if not within_rhigh, so throws away outer particles
            #equivalent to True & logical_not(True & True) = True & not(True) = True & False = False if distances > rlow and distances < rhigh
            #equivalent to True & not(False & True) = True & not(False) = True if distances <= rlow 
            # keep = evaluate("~inbin")     #but since I trim the particles so within_rhigh is trivially true (see above), this just reduces to not inbin, so no reason to calculate/store that     
        else:
            # within_rhigh = distances <= rhigh
            inbin = (distances > rlow) #&(within_rhigh)       #works for both 2D and 3D
            minsider = np.sum(masses)
            thism = np.sum(masses[inbin])
            # keep = within_rhigh & (~inbin)  #save logic as above

        m_of_r[ii] = thism
        density[ii] = thism/shell_vol

        distances = distances[~inbin]
        masses = masses[~inbin]
        positions = positions[~inbin]

        if pbar is not None:             pbar.update(ii)
    if pbar is not None:    pbar.finish()

    #swap everything back around so that I go from the inside out so that I can cumsum.  remember bins is already sorted because I didn't swap it; I created rev_bins.
    density = density[::-1]
    m_of_r = m_of_r[::-1]
    mltr = np.cumsum(m_of_r)

    #calculate keplerian circular velocity 
    vcirc = np.sqrt(G*mltr/bins)        #remember that bins didn't get reversed

    #remember this gets saved directly, so be good about naming!
    end = 'r'
    toreturn = {
            'density'             : density,
            'M.of.'+end           : m_of_r,
            'M.lt.'+end           : mltr,
            'vcirc'               : vcirc,
            'bins'                : bins,
                }

    return toreturn





def mass_profiles_nopair(bins, positions, masses, pair_distance, pbar_msg='Making mass profiles', nexpr=False):
    """
    computes: 
        M(<r), M(r), rho = M(r)/dV, Vcirc = sqrt(GM(<r)/r)
    
    assumes that particles closer to second host (whcih is pair_distance from main 
        host) are removed already.  removes the volume in that region from density 
        calculations.

    :bins : array-like : sorted (from small to large) bin edges to use
    :positions : array-like :  particle positions, with the center at 0,0,0
    :masses : array-like : particle masses, in the same order as positions and velocities
    :pbar_msg: str : what to print for the pbar (total mass and number of particles is appended)
    :nexpr : bool : whether or not to try to use numexpr to try to speed up the calculation
    """
    if nexpr:
        from numexpr import evaluate
        print("Using numexpr for the masking and summing masses")

    pair_midpoint_distance = pair_distance / 2.0


    #work from outside in, throwing away particles as I no longer need them
    assert positions.shape[0] == masses.shape[0]

    m_of_r = np.empty(bins.size)
    density = np.empty_like(m_of_r)
    distances = fast_dist(positions)            #center assumed to be at (0,0,0)

    #throw away any particles beyond my last bin edge
    msk = distances <= bins.max()
    positions = positions[msk]
    masses = masses[msk]
    distances = distances[msk]

    assert (np.sort(bins) == bins).all()        #make sure this is true because otherwise return will be nonsense since I use cumsum at the end
    rev_bins = bins[::-1]

    pbar_msg += '; Mtot(r < {:.0f} kpc)'.format(bins.max())
    pbar_msg += ' = {:.2g} Msun, {:,} particles)'.format(np.sum(masses),masses.size)

    for ii in tqdm(range(len(rev_bins)), pbar_msg):
        rhigh = rev_bins[ii]
        if ii == len(rev_bins)-1:
            rlow = 0
        else:
            rlow = rev_bins[ii+1]
        assert rlow < rhigh   

        if rhigh <= pair_midpoint_distance:
            shell_vol = 4./3.*np.pi*(rhigh**3 - rlow**3)
        else:
            #ok, more complicated because I need to subtract out the volume where the particles are trimmed
            # from wikipedia's article on spherical caps:
            #f the radius of the sphere is r and the height of the cap is h, then the volume of the spherical cap is:
            ### V= pi/3 * h^2 * (3r - h)
            cap_vol = lambda r, h: (np.pi/3.) * (h**2) * (3*r - h)

            if rlow <= pair_midpoint_distance:
                #then rhigh is over the border, but rlow is under it
                vol_low = 4./3. * np.pi * rlow**3
            else:
                height_of_low_cap = rlow - pair_midpoint_distance
                vol_of_low_cap = cap_vol(rlow, height_of_low_cap)
                low_vol_total = 4./3. * np.pi * rlow**3
                vol_low = low_vol_total - vol_of_low_cap

            height_of_high_cap = rhigh - pair_midpoint_distance
            vol_of_high_cap = cap_vol(rhigh, height_of_high_cap)
            vol_high_total = (4./3.) * np.pi * rhigh**3
            vol_high = vol_high_total - vol_of_high_cap
            shell_vol = vol_high - vol_low

        if nexpr:
            # within_rhigh = evaluate("(distances <= rhigh)")   #No need to do this -- I trim the particles before the loop and within the loop, so everything is within rhigh trivially
            minsider = evaluate("sum(masses)")
            inbin = evaluate("(distances > rlow)")
            thism = evaluate("sum(where(inbin,masses,0))")      #sum up the masses where inbin, 0 otherwise
            #particles that are within rhigh but not in the bin.  equivalent to  (within_rhigh) & (logical_not( (distances>rlow) & (within_rhigh) )
            #equivalent to False if not within_rhigh, so throws away outer particles
            #equivalent to True & logical_not(True & True) = True & not(True) = True & False = False if distances > rlow and distances < rhigh
            #equivalent to True & not(False & True) = True & not(False) = True if distances <= rlow 
            # keep = evaluate("~inbin")     #but since I trim the particles so within_rhigh is trivially true (see above), this just reduces to not inbin, so no reason to calculate/store that     
        else:
            # within_rhigh = distances <= rhigh
            inbin = (distances > rlow) #&(within_rhigh)       #works for both 2D and 3D
            minsider = np.sum(masses)
            thism = np.sum(masses[inbin])
            # keep = within_rhigh & (~inbin)  #save logic as above

        m_of_r[ii] = thism
        density[ii] = thism/shell_vol

        distances = distances[~inbin]
        masses = masses[~inbin]
        positions = positions[~inbin]

    #swap everything back around so that I go from the inside out so that I can cumsum.  remember bins is already sorted because I didn't swap it; I created rev_bins.
    density = density[::-1]
    m_of_r = m_of_r[::-1]
    mltr = np.cumsum(m_of_r)

    #calculate keplerian circular velocity 
    vcirc = np.sqrt(G*mltr/bins)        #remember that bins didn't get reversed

    #remember this gets saved directly, so be good about naming!
    end = 'r'
    toreturn = {
            'density'             : density,
            'M.of.'+end           : m_of_r,
            'M.lt.'+end           : mltr,
            'vcirc'               : vcirc,
            'bins'                : bins,
                }

    return toreturn











