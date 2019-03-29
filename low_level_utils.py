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

