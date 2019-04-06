#!/usr/bin/env


def snapfile_to_readargs(snapfile):
    """
    given the name of a snapshot file or 
    snapshot directory, return snapdir, 
    snapnum, and snapshot_name for the pfh
    readers
    """

    snapfile = snapfile.rstrip('/')
    if '/' not in snapfile:
        snapfile = './'+snapfile
    from os import path
    if path.isdir(snapfile):
        dodir = True
    else:
        dodir = False

    snapdir, snapshot_name = snapfile.rsplit('/', 1)
    snapshot_name, snapnum = snapshot_name.rsplit('_', 1)
    snapnum = int(snapnum.split('.')[0])

    if dodir:
        snapshot_name = 'snapshot'  # otherwise, tries to open the folder as a binary

    return snapdir, snapnum, snapshot_name


def read_part_wetzel(fname, types, assign_host_coordinates=False, subselect=False, quiet=False, **kwargs):
    """
    wrapper around :meth:`gizmo_analysis.gizmo_io.Read.read_snapshots` that takes 
    in a specific filename/snapshot  directory instead of the usual snapshot_values.

    useful for when the snapshots are unusual names or are in an unusual output 
    folder.  also makes sure that particle masses are always in float64, even if 
    everything else is 32 bit.  any kwargs are passed on to the 
    :meth:`gizmo_analysis.gizmo_io.Read.read_snapshots` function.

    finally, has support for subselecting to a sepecific region of space:
        if subselect is not False/None, then it should be a dictionary that contains
            * type  
                either "box" or "sphere" -- specifies whether you cut particles 
                within a sphere or particles within a box
            * radius 
                a number in the units of part that gives either half the box length 
                or the radius (depending on type)
            * center 
                a length 3 list-type that gives the center you want to select from.  
                if not given, then assign_host_coordinates must be true.
    
    """

    import gizmo_analysis as gizmo
    import utilities as ut
    from os import path, getcwd
    import numpy as np

    if subselect != None and subselect != False:
        assert type(subselect) == dict
        assert 'type' in subselect.keys()
        assert 'radius' in subselect.keys()
        if assign_host_coordinates == False:
            assert 'center' in subselect.keys()
            assert type(subselect['center']) == list or type(
                subselect['center']) == np.ndarray
            assert len(subselect['center']) == 3

    snapdir, snum, snapshot_name = snapfile_to_readargs(fname)
    if len(snapdir.split('/')) > 1:
        simulation_directory, snapshot_directory = snapdir.rsplit('/', 1)
        if not simulation_directory.startswith('/'):
            simulation_directory = getcwd() + '/' + simulation_directory
    else:
        simulation_directory = getcwd() + '/'
        snapshot_directory = snapdir

    if path.isfile(fname):
        myRead = gizmo.io.ReadClass(
            snapshot_name_base=snapshot_name.rsplit('_', 1)[0], quiet=quiet)
        part = myRead.read_snapshots(
            types, snapshot_values=snum, snapshot_value_kind='index',
            snapshot_directory=snapshot_directory, simulation_directory=simulation_directory,
            assign_host_coordinates=assign_host_coordinates, **kwargs)
    elif path.isdir(fname):
        myRead = gizmo.io.ReadClass(quiet=quiet)
        part = myRead.read_snapshots(
            types, snapshot_values=snum, snapshot_value_kind='index',
            snapshot_directory=snapshot_directory, simulation_directory=simulation_directory,
            assign_host_coordinates=assign_host_coordinates, **kwargs)
    else:
        raise IOError("{} doesn't exist".format(fname))

    if subselect:
        nkept = 0
        ntossed = 0
        if subselect['type'] == 'box':
            print(
                "Subselecting particles within a cube w/ half a side length of {}".format(subselect['radius']))
        else:
            print("Subselecting particles within a sphere of radius {}".format(
                subselect['radius']))

        if assign_host_coordinates:
            cen = part.host_positions[0]
        else:
            cen = subselect['center']

        r = subselect['radius']
        for k in part.keys():
            if subselect['type'] == 'box':
                msk = boxmsk(part[k]['position'], cen, r)
            else:
                msk = fast_dist(part[k]['position'], cen) <= r

            tn = np.count_nonzero(msk)
            nkept += tn
            ntossed += msk.shape[0] - tn

            for prop in part[k].keys():
                part[k][prop] = part[k][prop][msk]

    if 'convert_float32' in kwargs:
        # transform the mass back into float64 cause those have to be summed and that can cause problems
        for ptype in part:
            if 'mass' in part[ptype]:
                part[ptype]['mass'] = part[ptype]['mass'].astype(np.float64)
    return part


def readhead_wrapper(snapfile, verbose=False, docosmoconvert=1):
    """
    uses :meth:`gadget_lib.readsnap.readsnap` to read a gadget file header

    :param string snapfile:  The name of the snapshot/snapdir to read from
    :param bool verbose:  Print the parsed info or not
    :param bool docosmoconvert:  Passed to readsnap as cosmological, which 
        tells it to convert from cosmological (comoving) coordinates to 
        physical ones, and to deal with the h's as well

    Returns:
        dictionary read by readsnap with header information
    """

    from gadget_lib.readsnap import readsnap
    snapdir, snapnum, snapshot_name = snapfile_to_readargs(snapfile)
    if verbose:
        print("snapdir", snapdir)
        print("snapnum", snapnum)
        print("snapshot_name", snapshot_name)

    return readsnap(snapdir, snapnum, 1, snapshot_name=snapshot_name, h0=1, header_only=1, cosmological=docosmoconvert)


def readsnap_wrapper(snapfile, dmo, dogasT=False, verbose=False, docosmoconvert=1):
    """
    uses gadget_lib.readsnap.readsnap to read the header from a gadget files

    :param snapfile:  The name of the snapshot/snapdir to read from
    :param bool dmo:  Whether to read only the dark matter or not.
    :param bool dogasT:  whether or not to also return the gas temperature
    :param bool verbose:  Print the parsed info or not
    :param docosmoconvert:  Passed to readsnap as cosmological, which tells it to
        convert from cosmological (comoving) coordinates to physical ones, and to
        deal with the h's as well

    :returns:
        pos,vel, and mass for DM, gas, and stars in that order, then also gastemp if asked for
        always return gas and star data, but returns None if dmo is True
    """

    from gadget_lib.readsnap import readsnap
    snapdir, snapnum, snapshot_name = snapfile_to_readargs(snapfile)

    if verbose:
        print("snapdir", snapdir)
        print("snapnum", snapnum)
        print("snapshot_name", snapshot_name)

    if dmo:
        gpos, gvel, gmass, spos, svel, smass = None, None, None, None, None, None
        if dogasT:
            gastemp = None
    else:
        gas = readsnap(snapdir, snapnum, 0, snapshot_name=snapshot_name,
                       h0=1, cosmological=docosmoconvert)
        print("Read gas")
        star = readsnap(snapdir, snapnum, 4, snapshot_name=snapshot_name,
                        h0=1, cosmological=docosmoconvert)
        print("Read star")

        spos = star['p'][:]
        svel = star['v'][:]
        smass = star['m'][:]*1e10

        gpos = gas['Coordinates'][:]
        gvel = gas['Velocity'][:]
        gmass = gas['Masses'][:]*1e10
        if dogasT:
            gastemp = gas_temperature(gas['InternalEnergy'][:], gas['ne'][:])

    print("Starting dark read...")
    dark = readsnap(snapdir, snapnum, 1, snapshot_name=snapshot_name,
                    h0=1, cosmological=docosmoconvert)
    print("Read DM")
    dpos = dark['p'][:]
    dvel = dark['v'][:]
    dmass = dark['m'][:]*1e10

    if dogasT:
        # then gas temperature is expected.  if dmo, then I return None for it anyway
        return dpos, dvel, dmass, gpos, gvel, gmass, spos, svel, smass, gastemp
    else:
        # then gas temperature is (as default) not expected
        return dpos, dvel, dmass, gpos, gvel, gmass, spos, svel, smass
