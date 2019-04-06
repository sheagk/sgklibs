#!/usr/bin/env python3

def isint(string):
    try:
        int(string)
        return True
    except ValueError:
        return False

def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def normalize_array(ar):
    shifted_ar = ar - ar.min()
    if shifted_ar.max() == 0:
        return np.zeros(shifted_ar.shape)
    else:
        return shifted_ar/shifted_ar.max()


def fast_dist(positions,center=[0,0,0],nexpr=False):
    """
    uses numpy.subtract and numpy.linalg.norm to calculate
    distances quickly.  if nexpr is True and you have more 
    than one object, then it instead uses numexpr, which 
    might save some memory
    """
    
    if positions.shape == (3,):         #one object
        from numpy import linalg
        return linalg.norm(positions-center)
    elif nexpr:
        from numexpr import evaluate
        dist2 = evaluate("sum( (positions - center)**2, axis=1)")
        return evaluate("sqrt(dist2)")
    else:
        from numpy import subtract,linalg
        return linalg.norm(subtract(positions,center),axis=1)


def boxmsk(pos,cen,size):
    """
    returns a binary array that identifies points
    that fall within a box from -size to +size in
    each dimension

    :param pos:
        the points (3xN) to cut
    :param cen:
        the center of the box
    :param size:
        half the extent of the box
    """

    from numpy import array,abs
    px,py,pz = pos[:,0],pos[:,1],pos[:,2]
    x,y,z = cen
    return (abs(px-x)<size)&(abs(py-y)<size)&(abs(pz-z)<size)



def nearest_index(v1, v2):
    """
    find the index where an array is closest to a given value.
    can pass in the array/value in either order, but one must be
    a scalar and one must be an array.  

    returns np.argmin(np.abs(v1 - v2))
    """
    import numpy as np
    if np.isscalar(v1):
        assert not np.isscalar(v2)
    else:
        assert np.isscalar(v2)

    return np.argmin(np.abs(v1-v2))


def backup_file(fname):
    """
    if fname exists, then creates a copy of fname
    that's in a subfolder named 'bak' with a timestamp
    """
    import os,datetime,shutil
    if not os.path.isfile(fname):
        return
    now = datetime.datetime.now().isoformat().split('.')[0]
    if '/' not in fname:
        bakdir = 'bak/'
        obase = bakdir + fname
    else:
        bakdir = fname.rsplit('/',1)[0]+'/bak/'
        obase = bakdir + fname.rsplit('/',1)[-1]
    if not os.path.isdir(bakdir):
        os.mkdir(bakdir)
    if '.' not in obase:
        shutil.copyfile(fname,obase+now)
    else:
        shutil.copyfile(fname,obase.rsplit('.',1)[0]+now+'.'+obase.rsplit('.',1)[-1])


def sanitize_order(order):
    """
    takes an array of values (intended to be some sort of order/indexing thing) 
    and makes it start at zero and be sequential with no duplicates.


    gaps are eliminated, and ties are decided based on of appearance in list

    negative numbers are removed
    """

    import numpy as np

    order = np.array(order)
    order = order[order>=0]

    order -= order.min()

    #first handle non-uniques by incrementing all values greater than each non-unique one
    uniq_order, uniq_indices, uniq_counts = np.unique(order, return_index=True, return_counts=True)
    for idx, num in enumerate(order):
        if idx not in uniq_indices:
            order[order>=num] = order[order>=num] + 1
            #but remember to then decrement the first instance of this number
            to_dec = np.where(order==num+1)[0][0]
            order[to_dec] = order[to_dec] - 1

    #ok, now sort, then handle gaps:
    sorti = np.argsort(order)
    un_sorti = np.argsort(sorti)    #use this as an indexer to go back to the order passed in
    sorted_order = order[sorti]

    #now handle gaps:
    for idx, num in enumerate(sorted_order):
        if idx == 0:
            continue

        if sorted_order[idx] != sorted_order[idx-1] + 1:
            diff = (sorted_order[idx] - sorted_order[idx-1]) - 1
            assert diff > 0
            msk = sorted_order >= sorted_order[idx]
            sorted_order[msk] = sorted_order[msk] - diff

    return sorted_order[un_sorti]


def get_midpoints(ar, mode='linear'):
    """
    Returns the midpoints of an array; i.e. if you have the left edge of a set
    of bins and want the middle, this will do that for you.

    :param ar:
        The array or list of length L to find the midpoints of
    :param mode:
        Whether to find the midpoint in logspace ('log') or linear
        space ('linear')

    :returns:
        An array of the midpoints of length L - 1
    """
    _valid_modes = ['linear', 'log']
    if mode not in _valid_modes:
        raise TypeError("Unrecognize midpoint method; must be one of {}}.".format(
            _valid_modes))
    if mode == 'linear':
        from numpy import array
        lst = [ar[i] + (ar[i+1]-ar[i])/2 for i in range(len(ar)) if i != len(ar) -1]
        return array(lst)
    elif mode == 'log':
        from numpy import array,log10
        lst = [10**(log10(ar[i]) + (log10(ar[i+1])-log10(ar[i]))/2) for i in range(len(ar)) if i != len(ar) -1]
        return array(lst)
    else:
        raise TypeError("How did I get here?  provided mode = {}".format(mode))


def format_sci(num,precision=2):
    '''
    given a number, returns a string that 
    represents that number in scientific notation with
    the specified precision using latex
    '''

    from numpy import log10,abs
    from math import floor
    num = float(num)
    if num == 0:
        return r'$0$'
    power = int(floor(log10(num)))
    leftover = num/10**power
    if leftover == 1 or abs(leftover - 1.0) < 1e-10:
        return r'$10^{'+str(power)+'}$'
    else:
        if round(leftover,precision) == int(leftover):
            toreturn = '$'+str(int(leftover))+r'\times 10^{'+str(power)+'}$'
        else:
            toreturn = '$'+str(round(leftover,precision))+r'\times 10^{'+str(power)+'}$'
        toreturn = toreturn.replace(r'.0\times',r'\times')
        if toreturn.startswith(r'1\times'):
            toreturn = toreturn[len(r'1\times'):]
        return toreturn


def extract_positions_masses(part, species='all', weight='mass'):
    '''
    given a particle dictionary, returns an array of the positions
    and (by default) masses flattened over the species.  convenient
    for when you want to make a density plot, since this is what you
    need for that. 
    '''
    species = ut.particle.parse_species(part, species)
    npart = np.sum([part[spec]['mass'].size for spec in species])

    positions = np.empty((npart, 3))
    masses = np.empty(npart)

    left = 0
    for spec in species:
        right = left + part[spec]['mass'].size

        positions[left:right] = part[spec]['position']
        masses[left:right] = part[spec].prop(weight)

        left = right
    return positions, masses
