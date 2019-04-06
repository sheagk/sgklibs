#!/usr/bin/env python3

from __future__ import division, print_function

import os
import h5py
import numpy as np
import utilities as ut
import gizmo_analysis as gizmo_analysis
import rockstar_analysis as rockstar
from astropy.cosmology import LambdaCDM
from copy import copy

from .low_level_utils import isfloat, isint, nearest_index
from .particle_io import read_part_wetzel


class Galaxy():
    def __init__(
            self,
            hal=None, index=None,
            part=None,
            halt=None, halt_index=None,
            arbor=None, arbor_root_index=None,
            hals=None,
            fname=None,
            host2=False, **kwargs):
        """
        general galaxy class, work in progress (as is everything)
        pass in either a galaxy/halo dictionary (or a path to one) and an 
        index in that dictionary to identify a galaxy AND/OR pass in 
        a ParticleData dictionary (or a path to a snapshot file).  best
        if both are given, but if only the ParticleData is available, then
        you can also pass in a "center_position" as a kwarg to define the
        center and most other stuff will be figured out as best as possible.

        Note that if you pass in *only* part (w/o a center position), then
        the class will assume that you want the largest object, and it'll do
        the standard iterative zoom centering algorithm

        you can also pass in a pickled file as fname and load an existing galaxy
        class (w/o the particle data and halo catalog).  the code will do this AFTER
        loading the halo catalog and particle data (if those are also passed in),
        so anything created should be saved

        :param hal: 
            either a HaloDictionary (loaded via rockstar.rockstar_io.IO.read_catalogs)
            or a path to a file to read with that function (arguments related to id-ing 
            the file will be figured out, but can pass in extra ones via kwargs)

        :param index: 
            index of this galaxy in the halo/galaxy catalog

        :param part:
            either a ParticleDictionary (loaded via gizmo.gizmo_io.Read.read_snapshots,
            or a wrapper around it), or a path to a file to be read with that function
            (arguements related to id-ing the file are again figured out, and there are
            defaults for most other arguements included, but pass in any extra via kwargs)

        :param halt:
            either: 
                * a HaloDictionary that holds a merger tree (loaded via rockstar.rockstar_io.IO.read_tree) 
                * an empty string to load with the defaults
                * a non-empty string to pass as `rockstar_directory` to that function

        :param fname:
            an instance of the Galaxy class saved with Galaxy.save 

        :param host2:
            if this is True, then you can pass in just hal (w/o an index), and you
            get the halo that rockstar.io decides is the ''secondary host'' (i.e.
            second most massive halo that's not overly contaminated)

        :param kwargs:
            lots and lots of stuff can be passed in here.  important ones are below:

            :param two_hosts:
                specify that a catalog should be read as if it's a local group.  only 
                matters if reading from an ascii, list, or ahf file (i.e., if it's not
                already in the HaloDictionary that's passed in the HDF5 file that it's 
                told to load, it won't add it).  HDF5 halo catalogs should already have 
                host2.distance or another similar host2 quantity, which will be used to 
                define whether or not there are two hosts.

            :param distance_max:
                maximum distance to use to define the galaxy (passed to get_galaxy_properties)
                if no halo catalog is passed in.

            :param species:
                particle species to read, if you pass in a path to a file as part
                instead of the particles themselves

            :param properties:
                particle properties to read, if reading particles

            :param center_position:
                center_position (if not passing in a halo catalog) if you don't want the
                code to identify the largest object in the box

            :param center_velocity:
                center velocity (if not passing in a halo catalog) if you don't want the
                code to calculate it

        """
        self.center_position = None
        self.center_velocity = None
        self.index = None
        self.halt_index = None
        self.hals = None
        self.host2 = False
        self.arbor = None
        self.arbor_root = None
        self.arbor_root_index = None

        self.halo_file = None
        self.particle_file = None
        self.galaxy_properties = None
        self.halo_properties = None
        self.star_indices = None
        self.host_index = None  # index of the halo that hosts this one, if there is one
        self.host_distance = None  # distance to the halo that hosts this one, if there is one
        self.host_distance_total = None
        self.is_primary = None
        self.is_satellite = None
        self.is_field = None
        self.type = None  # host, satellite, or field
        self.halo_mass = None
        self.main_branch_indices = None
        self.hals_mb_indices = None

        # don't calcualte these until they're asked for (they get stored in those functions)
        self.sfh_moments = None
        self.sfh = None
        self.sfr = None
        self.burstiness = None
        self.prj = None

        # need to have one of the following for this galaxy to be worth creating
        assert (hal is not None) or (fname is not None) or (halt is not None) or (
            part is not None) or (hals is not None) or (arbor is not None)

        # index can either be the index to hal or the index to hals[-1]
        final_hals_index = copy(index)

        # clear out whichever it isn't
        if hal is None:
            index = None
        if hals is None:
            final_hals_index = None

        # assign the kwargs themselves and a few rudimentary properties from the kwargs
        self.kwargs = kwargs

        self.two_hosts = False  # default to this...
        if self.kwargs.get('two_hosts', False):
            # print("setting two hosts manually")
            self.two_hosts = True
        elif host2:
            # print("working with host2, so setting two hosts")
            self.two_hosts = True
        elif 'elvis' in os.getcwd():
            # print("found elvis in cwd, so setting two_hosts")
            self.two_hosts = True

        if host2:
            self.host2 = True
            self.kwargs['is_primary'] = True
            self.kwargs['two_hosts'] = True

        self.simulation = kwargs.get('simulation', '')
        self.simulation_directory = kwargs.get('simulation_directory', '.')

        # read in the halo data, if I need to
        self.assign_hal(hal, index)

        # read / verify the halo merger tree:
        self.assign_halt(halt, halt_index)

        # read and verify the particle data
        self.assign_part(part)

        # read and verify the ytree arbor, and either store the index or compute it
        self.assign_arbor(arbor, arbor_root_index)

        # read and verify the list of halo catalogs (i.e. hals)
        self.assign_hals(hals, final_hals_index)

        # either load up the rest of the initialization data (and any other properties calcualted before the Galaxy was saved)
        if fname is not None:
            if not os.path.isfile(fname):
                raise IOError(
                    "Asked to load galaxy class from {}, but that file doesn't exist".format(fname))
            self.load(fname)

        # or assign those properties from the hal, halt, and part that were passed in
        # do not do the assignment if fname is passed in, because hal, halt, and part are frequently not loaded again
        else:
            # assign center position:
            self.assign_center_position_velocity()

            # assign what type (satellite/host/field) I'm looking at
            self.assign_type()

            # assign star indices:
            self.assign_indices('star')

            # assign masses:
            self.assign_mass_radius()

            # assign info etc.
            self.assign_info_etc()

        # if I get this far, I need to at least have a center position
        assert self.center_position is not None

    def assign_hal(self, hal, index):
        if hal is not None:
            assert (index is not None) or (
                'center_position' in self.kwargs) or (self.host2)
        if index is not None:
            assert hal is not None

        if hal is not None:
            if type(hal) == str:
                if os.path.isfile(hal):
                    hal = self.load_hal(hal)
                else:
                    raise IOError(
                        "Cannot find {} to read halo catalog".format(hal))
            else:
                assert type(hal) == rockstar.rockstar_io.HaloDictionaryClass
        self.hal = hal

        # assign the index of this galaxy
        if index is not None:
            self.index = int(index)
        elif self.hal is not None:
            # hal is not none => can assign index even if I don't have it passed in because I asserted that
            if self.host2:
                if 'host2.distance' not in self.hal:
                    rockstar.io.IO._assign_host_to_catalog(
                        self.hal, host_kind='star',  host_number=2)
                self.index = np.argmin(self.hal.prop('host2.distance.total'))
            else:
                # index is None and hal is not None => assign index based on center_position (unless looking for the second most massive host)
                # note that I require center_position to be in kwargs in this case (assertions above)

                # set the index as the halo closest to passed in center
                self.index = np.argmin(np.linalg.norm(ut.coordinate.get_distances(
                    hal.prop('position'), kwargs['center_position'])))
                print("-- using halo {}, which is {} kpc from the passed in center_position".format(
                    self.index, ut.coordinate.get_distances(hal.prop('position')[self.index], kwargs['center_position'])))

    def assign_halt(self, halt, halt_index):
        if halt is not None:
            # need one of these for the halo tree to make sense
            assert (self.hal is not None) or (halt_index is not None)

        if halt is not None:
            if type(halt) == str:
                halt = self.load_halt(halt)
            else:
                assert type(halt) == rockstar.rockstar_io.HaloDictionaryClass
        self.halt = halt

        if self.hal is not None and self.halt is not None:
            if halt_index is not None:
                # index should correspond to the tree.index in the catalog
                assert halt_index == self.hal_prop('tree.index')
            # already checked for agreement with passed in value, so just grab from the tree
            self.halt_index = self.hal_prop('tree.index')
            if self.halt_index < 0:
                # passed in a phantom; double-check it is a phantom -- TODO this is under development
                assert self.hal_prop('progenitor.number') <= 0
                print("!! -- warning:  passed in a tree, a catalog, and a catalog index ({}), but that catalog index corresponds to a phantom halo with no progenitors.  will ignore the tree".format(self.index))
                self.halt = None
                self.halt_index = None
            else:
                # should be able to go back and forth if not a phantom
                assert self.halt.prop('catalog.index')[
                    self.halt_index] == self.index
        elif self.halt is not None:
            # don't have hal, but do have halt, so must have handed in a halt index
            self.halt_index = halt_index

    def assign_part(self, part):
        if part is not None:
            if type(part) == str:
                if os.path.isfile(part) or os.path.isdir(part):
                    part = self.load_part(part)
                else:
                    raise IOError(
                        "Cannot find {} to read particle data".format(part))
            else:
                assert type(part) == ut.basic.array.DictClass
        self.part = part

    def assign_arbor(self, arbor, root_index):
        if arbor is not None:
            import ytree
            if type(arbor) == str:
                arbor = ytree.load(arbor)
            else:
                assert type(type(arbor)) == ytree.arbor.arbor.RegisteredArbor

            arbor.add_alias_field('vel.circ.max', 'Vmax', 'km/s')
            arbor.add_alias_field('radius', 'virial_radius', 'kpc')

            if root_index is None:
                # if not passing in an index for the root, then you must have a halo catalog and index there to match
                assert (self.hal is not None and self.index is not None)

                distances = ut.coordinate.get_distances(self.hal_prop(
                    'position'), self.arbor['position'], total_distance=True)
                sorti = np.argsort(np.abs(self.hal_prop(
                    'vel.circ.max') - self.arbor['vel.circ.max']))
                root_index = sorti[distances < 10][0]

            self.arbor_root = arbor[root_index]

        self.arbor = arbor
        self.arbor_root_index = root_index

    def assign_hals(self, hals, final_hals_index):
        if hals is not None:
            if type(hals) == list:
                assert hals[-1].info['catalog.kind'] == 'halo.catalog'
                self.hals = hals
            elif type(hals) == str:
                # try to read in the halo catalogs; assume passed in param is rockstar_directory
                print(
                    "Reading out.list catalogs assuming rockstar_directory = {}".format(hals))
                self.hals = rockstar.io.IO.read_catalogs(
                    snapshot_value_kind='index', snapshot_values=None, rockstar_directory=hals, file_kind='out')
            elif type(hals) == dict:
                print("Reading halo catalogs using arguments supplied.")
                snapshot_value_kind = hals.get('snapshot_value_kind', 'index')
                snapshot_values = hals.get('snapshot_values', None)
                rockstar_directory = hals.get(
                    'rockstar_directory', 'halo/rockstar_dm/')
                file_kind = hals.get('file_kind', 'out')
                self.hals = rockstar.io.IO.read_catalogs(snapshot_values=snapshot_values,
                                                         snapshot_value_kind=snapshot_value_kind, file_kind=file_kind, rockstar_directory=rockstar_directory)
            else:
                print(
                    "Passed in hals that is neither a string nor list; assuming you want to load all catalogs with defaults")
                self.hals = rockstar.io.IO.read_catalogs(
                    snapshot_value_kind='index', snapshot_values=None)

        if self.hals is None:
            # no need to handle final_hals_index if no hals
            return

        if self.hal is None:
            print("Assigning last halo catalog as hal")
            self.hal = self.hals[-1]
        else:
            # make sure we're looking at the same halo catalog by comparing mass arrays
            hal_mass = self.hal.prop('mass')
            last_hals_mass = self.hals[-1].prop('mass')

            assert last_hals_mass.size == hal_mass.size
            assert np.max(np.abs(hal_mass - last_hals_mass) /
                          (0.5*(hal_mass + last_hals_mass))) < 0.01

        if (self.index is None) and (final_hals_index is not None):
            self.index = final_hals_index
        elif (final_hals_index is None) and (self.index is not None):
            assert np.abs(self.hals[-1].prop('mass', self.index) - self.hal_prop('mass'))/(
                0.5*(self.hals[-1].prop('mass', self.index)+self.hal_prop('mass'))) < 0.05
        elif self.index != final_hals_index:
            raise Warning(
                "!!! somehow got different values for index and final_hals_index; may get undefined behavior !!!")
        elif (self.index is None) and (final_hals_index is None) and (self.is_primary or self.host2):
            # can assign an index if I'm looking at a host
            print("Assigning index via hostX.distance.total in last halo catalog")
            if self.is_primary:
                self.index = np.argmin(
                    self.hals[-1].prop('host.distance.total'))
            elif self.host2:
                self.index = np.argmin(
                    self.hals[-1].prop('host2.distance.total'))

        return

    def assign_center_position_velocity(self):
        # assign from hardcoded center (that isn't None)
        if 'center_position' in self.kwargs and self.kwargs.get('center_position') is not None:
            self.center_position = self.kwargs['center_position']

            if 'center_velocity' in self.kwargs:
                self.center_velocity = self.kwargs['center_velocity']
            elif self.hal is not None:
                vel_prop = 'star.velocity' if 'star.velocity' in self.hal else 'velocity'
                self.center_velocity = self.hal_prop(vel_prop)
            elif self.halt is not None:
                vel_prop = 'star.velocity' if 'star.velocity' in self.halt else 'velocity'
                self.center_velocity = self.halt_prop(vel_prop)
            elif self.part is not None:
                self.center_velocity = ut.particle.get_center_velocities(
                    self.part, center_positions=np.array([self.center_position]))

        # assign from the halo catalog
        elif self.hal is not None:
            if 'star.position' in self.hal:
                self.center_position = self.hal_prop('star.position')
                self.center_velocity = self.hal_prop('star.velocity')
            else:
                self.center_position = self.hal_prop('position')
                self.center_velocity = self.hal_prop('velocity')

        # assign from the halo tree
        elif self.halt is not None:
            if 'star.position' in self.halt:
                self.center_position = self.halt_prop('star.position')
                self.center_velocity = self.halt_prop('star.velocity')
            else:
                self.center_position = self.halt_prop('position')
                self.center_velocity = self.halt_prop('velocity')

        # assign the most massive object as a last resort from the particle data
        elif self.part is not None:
            print("!! -- no halo catalog and no center position passed in")
            print("!! -- centering iteratively on the biggest galaxy in the simulation")

            gizmo_analysis.gizmo_io.Read.assign_host_coordinates(self.part)
            self.center_position = self.part.host_positions[0]
            self.center_velocity = self.part.host_velocities[0]
            self.kwargs['is_primary'] = True

    def assign_type(self):
        if self.kwargs.get('is_primary', False):
            self.type = 'host'
            self.is_primary = True
            self.is_satellite = False
        elif self.kwargs.get('is_satellite', False):
            self.type = 'satellite'
            self.is_primary = False
            self.is_satellite = True
        elif self.kwargs.get('is_field', False):
            self.type = 'field'
            self.is_primary = False
            self.is_satellite = False
        elif self.hal is not None:
            self._assign_type_from_hal()
        elif self.halt is not None:
            self._assign_type_from_halt()
        elif self.part is not None:
            self._assign_type_from_part()
        else:
            print("!! -- no way to assign host status!  assuming primary.")
            self.type = 'host'
            self.is_primary = True
            self.is_satellite = False

    def assign_indices(self, spec='star', verbose=1):
        assigned = False
        if self.hal is not None:
            if spec+'.indices' in self.hal:
                if verbose > 1:
                    print("-- assigning "+spec+"_indices from hal")
                self.__dict__[spec+'_indices'] = self.hal_prop(spec+'.indices')
                assigned = True
        if not assigned and self.halt is not None:
            if spec+'.indices' in self.halt:
                if verbose > 1:
                    print("-- assigning "+spec+"_indices from halt")
                self.__dict__[
                    spec+'_indices'] = self.halt_prop(spec+'.indices')
                assigned = True
        if not assigned and self.part is not None:
            # use get_galaxy_properties or get_halo_properties to find indices
            if spec in ['star', 'gas']:
                if verbose > 0:
                    print(
                        "-- assigning {} indices via get_galaxy_properties".format(spec))
                gal_prop = self.get_galaxy_properties(species_name=spec)
                self.__dict__[spec+'_galaxy_properties'] = gal_prop
                self.__dict__[spec+'_indices'] = gal_prop['indices']
                assigned = True
            elif spec == 'dark':
                if verbose > 0:
                    print(
                        "-- assigning {} indices via get_halo_properties".format(spec))
                halo_prop = self.get_halo_properties(distance_min=0.01)
                self.halo_properties = halo_prop
                self.dark_indices = halo_prop['indices']
                assigned = True
        if not assigned:
            print("-- !! warning: no way to get indices for {}".format(spec))

    def assign_mass_radius(self):
        self.galaxy_mass = self.get_galaxy_mass()
        self.galaxy_radius = self.get_galaxy_radius()
        if (not self.is_satellite) or (self.hal is not None):
            # can only assign a halo mass if I'm not a satellite -- virial criteria doesn't make sense otherwise -- or if I have a halo catalog to draw from
            self.halo_mass = self.get_halo_mass()
            self.halo_radius = self.get_halo_radius()

    def assign_info_etc(self):
        if self.hal is not None:
            self.Cosmology = self.hal.Cosmology
            self.info = self.hal.info
            self.snapshot = self.hal.snapshot
            self.Snapshot = self.hal.Snapshot
            self.Cosmology = self.hal.Cosmology
            self.element_dict = self.hal.element_dict
            self.element_pointer = self.hal.element_pointer
        elif self.halt is not None:
            self.Cosmology = self.halt.Cosmology
            self.info = self.halt.info
            self.snapshot = self.halt.snapshot
            self.Snapshot = self.halt.Snapshot
            self.Cosmology = self.halt.Cosmology
            self.element_dict = self.halt.element_dict
            self.element_pointer = self.halt.element_pointer
        elif self.part is not None:
            self.info = self.part.info
            self.snapshot = self.part.snapshot
            self.Snapshot = self.part.Snapshot
            self.Cosmology = self.part.Cosmology

            if 'star' in self.part:
                element_dict = self.part['star'].element_dict
                element_pointer = self.part['star'].element_pointer
            elif 'gas' in self.part:
                element_dict = self.part['gas'].element_dict
                element_pointer = self.part['gas'].element_pointer

        if self.Cosmology is not None:
            self.Cosmo = LambdaCDM(
                H0=100*self.Cosmology['hubble'],
                Om0=self.Cosmology['omega_matter'],
                Ob0=self.Cosmology.get('omega_baryon'),
                Ode0=self.Cosmology['omega_lambda']
            )

    def save(self, outname, skip=['part', 'hal', 'prj', 'halt', 'hals', 'arbor']):
        # skip is the list of things NOT to save because they're prohibitively large

        import pickle
        tmp_dict = dict([(k, self.__dict__[k])
                         for k in self.__dict__ if k not in skip])
        with open(outname, 'wb') as f:
            pickle.dump(tmp_dict, f)

    def load(self, inname):
        import pickle
        with open(inname, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)

    def load_hal(self, hal):
        if '/' not in hal:
            hal = './'+hal

        self.halo_file = hal
        IO = rockstar.io.IO

        file_kind = hal.split('.')[-1]
        if hal.count('/') > 1:
            directory = hal.rsplit('/', 2)[0]
            catalog_directory = hal.rsplit('/', 2)[1]+'/'
        elif hal.count('/') == 1:
            directory = './'
            catalog_directory = hal.rsplit('/', 1)[0]+'/'
        else:
            directory = './'
            catalog_directory = '/'

        if file_kind == 'AHF_halos':
            file_kind = 'ahf'
            snapshot_value_kind = 'redshift'
            snapshot_values = redshift = float(
                hal.split('.z')[-1].split('.AHF')[0])

            # rockstar_directory = 'NULL'
            rockstar_directory = directory
            IO.catalog_directory = catalog_directory

        else:
            snapshot_value_kind = 'index'
            snapshot_values = 600
            for possible_number in hal.split('/')[-1].rsplit('.')[0].split('_'):
                if isint(possible_number):
                    snapshot_values = int(possible_number)
                    break

            # ahf_directory = 'NULL'
            rockstar_directory = directory
            if file_kind == 'hdf5':
                IO.catalog_hdf5_directory = catalog_directory

            else:
                IO.catalog_directory = catalog_directory
                if file_kind == 'list':
                    if 'hlist' in hal:
                        file_kind = 'hlist'
                    else:
                        file_kind = 'out'

        print("-- loading halos from {} as {} with {} = {}".format(self.halo_file,
                                                                   file_kind, snapshot_value_kind, snapshot_values))

        # IO.two_hosts = self.two_hosts   # = kwargs.get('two_hosts', False)
        return IO.read_catalogs(snapshot_value_kind=snapshot_value_kind, snapshot_values=snapshot_values,
                                rockstar_directory=rockstar_directory, file_kind=file_kind, simulation_directory=self.simulation_directory,
                                assign_species=self.kwargs.get('assign_species', True), assign_host=self.kwargs.get('assign_host', True))

    def load_halt(self, halt):
        IO = rockstar.io.IO
        # IO.two_hosts = self.two_hosts  # = kwargs.get('two_hosts', False)
        if not len(halt):
            self.halt_directory = 'halo/rockstar_dm'
            return IO.read_tree()
        else:
            if halt.endswith('tree.hdf5'):
                self.halt_directory = halt.rsplit('/', 2)[0]
            else:
                self.halt_directory = halt
            return IO.read_tree(rockstar_directory=self.halt_directory)

    def load_part(self, part):
        self.particle_file = part

        species = self.kwargs.get('species', 'star')
        properties = self.kwargs.get('properties', [
                                     'position', 'velocity', 'mass', 'id', 'form.scalefactor', 'massfraction'])
        # by default, just load total metallicity
        element_indices = self.kwargs.get('element_indices', 0)
        convert_float32 = self.kwargs.get('convert_float32', True)
        element_indices = self.kwargs.get('element_indices', [0, 1])
        subselect = self.kwargs.get('subselect', False)

        if subselect:
            assert type(subselect) == dict and self.hal is not None
            subselect['center'] = subselect.get('center', self.center_position)

        return read_part_wetzel(part, species,
                                properties=properties, convert_float32=convert_float32,
                                element_indices=element_indices, subselect=subselect)

    def get_halo_mass(self):
        # get the halo mass of the galaxy in a few ways:
        # 0. look for an existing halo_properties
        # 1. try getting an entry in the halo (need hal)
        # 2. try running get_halo_properties
        # 3. give up!  can't do much without either part or hal
        if self.halo_properties is not None:
            return self.halo_properties['mass']
        if self.hal is not None:
            return self.hal_prop('mass')
        if self.halt is not None:
            return self.halt_prop('mass')
        if self.part is not None:
            if 'dark' in self.part:
                self.halo_properties = self.get_halo_properties()
                return self.halo_properties['mass']
        return None

    def get_halo_radius(self):
        # get the halo radius of the galaxy in a few ways:
        # 0. look for an existing halo_properties
        # 1. try getting an entry in the halo (need hal)
        # 2. try running get_halo_properties
        # 3. give up!  can't do much without either part or hal
        if self.halo_properties is not None:
            return self.halo_properties['radius']
        if self.hal is not None:
            return self.hal_prop('radius')
        if self.halt is not None:
            return self.halt_prop('radius')
        if self.part is not None:
            if 'dark' in self.part:
                self.halo_properties = self.get_halo_properties()
                return self.halo_properties['radius']
        return None

    def get_galaxy_mass(self):
        # get the stellar mass of the galaxy in a few ways:
        # 1. try summing up the masses of the stars (need part)
        # 2. try getting an entry in the halo (need hal)
        # 3. give up!  can't do much without either of those
        if self.part is not None:
            return self.star_particle_prop('mass').sum()
        if self.hal is not None:
            if 'star.mass' in self.hal:
                return self.hal.prop('star.mass')[self.index]
        return None

    def get_galaxy_radius(self, galaxy_prop=None):
        # get the radius of the galaxy in a few ways:
        # 0. load it from the passed in galaxy_prop
        # 1. try looking for a self.galaxy_properties
        # 2. try looking in the halo catalog
        # 3. try running ut.particle.get_galaxy_properties with self.part_indices
        # 4. give up!  can't do much without either a galaxy prop, the particle data, or the halo catalog
        if galaxy_prop is not None:
            return galaxy_prop['radius']
        if self.galaxy_properties is not None:
            return self.galaxy_properties['radius']

        ev = self.kwargs.get('edge_value', 90)
        if self.hal is not None:
            if 'star.radius.'+str(ev) in self.hal:
                return self.hal.prop('star.radius.'+str(ev))[self.index]
        if self.part is not None:
            galaxy_props = self.get_galaxy_properties(
                species_name='star', part_indices=self.star_indices, edge_value=ev)
            self.galaxy_properties = galaxy_props
            return self.galaxy_properties['radius']
        return None

    def hal_prop(self, prop=''):
        assert self.hal is not None
        return self.hal.prop(prop, self.index)

    def halt_prop(self, prop=''):
        assert self.halt is not None
        return self.halt.prop(prop, self.halt_index)

    def star_particle_prop(self, prop=''):
        assert self.part is not None
        if 'distance' in prop:
            return ut.coordinate.get_distances(
                self.star_particle_prop('position'), self.center_position,
                periodic_length=self.info['box.length'], total_distance='total' in prop)
        return self.part['star'].prop(prop)[self.star_indices]

    def _assign_type_from_hal(self):  # , virial_kind='vir'):
        '''
        assign 
            self.is_primary
            self.is_satellite
            self.is_field
            self.host_index
            self.host_distance
            self.host_distance_total
            self.host2_distance
            self.host2_distance_total
        '''
        assert self.hal is not None
        if 'host2.distance' in self.hal and self.two_hosts == False:
            self.two_hosts = True
            # print("found host2.distance in hal, so setting two_hosts")

        if 'host.distance' not in self.hal:
            rockstar.io.IO._assign_host_to_catalog(self.hal, 'star')
        if self.two_hosts and 'host2.distance' not in self.hal:    # i.e. if I assigned two hosts another way
            rockstar.io.IO._assign_host_to_catalog(
                self.hal, 'star', host_number=2)

        self.host_distance = self.hal_prop('host.distance')
        self.host_distance_total = self.hal_prop('host.distance.total')
        self.host_index = self.hal['host.index'][0]

        if self.two_hosts:
            self.host2_distance = self.hal_prop('host2.distance')
            self.host2_distance_total = self.hal_prop('host2.distance.total')
            self.host2_index = self.hal['host2.index'][0]
        else:
            self.host2_distance_total = np.inf
            self.host2_distance = np.array([np.inf, np.inf, np.inf])
            self.host2_index = -2**31

        if self.host_distance_total == 0:
            self.is_satellite = False
            self.is_primary = True
            self.is_field = False
            self.type = 'host'
            return

        if self.host2_distance_total == 0:
            self.type = 'host'
            self.is_satellite = False
            self.is_primary = True
            self.is_field = False
            return

        if self.host_distance_total <= self.hal.prop('radius')[self.host_index]:
            self.is_satellite = 'host1'
            # self.host_index = host_index
            self.is_primary = False
            self.is_field = False
            self.type = 'satellite'
            return

        if self.two_hosts:
            if (self.host2_distance_total <= self.hal.prop('radius')[self.host2_index]):
                self.type = 'satellite'
                self.is_satellite = 'host2'
                self.is_primary = False
                self.is_field = False
                return

        # if i get this far, then I'm not (either) primary host or a satellite of either
        self.type = 'field'
        self.is_satellite = False
        self.is_primary = False
        self.is_field = True
        return

    def _assign_type_from_part(satellite_distance_cut=300, host_distance_cut=1.0):
        assert self.part is not None
        if self.two_hosts:
            raise Warning(
                "!! -- identifying type via particle data, which will miss (satellites of) the second host!")

        if 'dark' in self.part:
            print("-- finding center of largest halo to find out if I'm a satellite...")
            gizmo_analysis.gizmo_io.Read.assign_center(
                self.part, species_name='dark')
            host_center_position = np.array(
                self.part.center_position, copy=True)
            host_center_velocity = np.array(
                self.part.center_velocity, copy=True)
            halo_properties = self.get_halo_properties(
                center_position=host_center_position, virial_kind=virial_kind)
            host_virial_radius = halo_properties['radius']

        else:
            from .astro_relationships import abundance_match_behroozi_2012, bryan_norman_virial_ciritcal_overdensity
            print(
                "-- finding center of largest galaxy to approximate satellite status...")
            gizmo_analysis.gizmo_io.Read.assign_center(self.part)
            host_center_position = np.array(
                self.part.center_position, copy=True)
            host_center_velocity = np.array(
                self.part.center_velocity, copy=True)

            if satellite_distance_cut is None or satellite_distance_cut <= 0:
                host_galaxy_properties = self.get_galaxy_properties(
                    center_position=host_center_position)
                print(
                    "-- crudely approximating virial radius from stellar mass via Behroozi+2013...")
                host_mstar = host_galaxy_properties['mass']

                # strictly speaking, the below block is wrong because of the fact that
                # median mstar at a given mvir (which is what I calculate) can't be
                # directly inverted to give you median mvir at a given mstar (cause of
                # the way  errors work).

                # but fuck it, this is all extremely approximate anyway

                mvir_bins = np.logspace(8, 13, 1e3)
                mstar_bins = abundance_match_behroozi_2012(
                    mvir_bins, 0, alpha=-1.92)
                # interpolate where mvir = host_mstar
                host_mvir_approx = np.interp(host_mstar, mstar_bins, mvir_bins)
                density_threshold = bryan_norman_virial_ciritcal_overdensity(
                    self.snapshot['redshift'], self.Cosmo.Om0, self.Cosmo.Ode0)
                host_virial_radius = (3.0 * host_mvir_approx / (4*np.pi * self.Cosmo.critical_density0.to(
                    'Msun / kpc^3').value * density_threshold))**(1./3.)
                print("-- got Mstar = {0:.2g} Msun => Mvir = {1:.2g} Msun => Rvir = {2:.0f} kpc".format(
                    host_mstar, host_mvir_approx, host_virial_radius))
            else:
                host_virial_radius = satellite_distance_cut

        self.host_distance = self.center_position - host_center_position
        self.host_distance_total = np.linalg.norm(self.host_distance)

        self.is_satellite = False
        self.is_field = False
        self.is_primary = False
        if self.host_distance_total <= host_distance_cut:
            self.is_primary = True
            self.type = 'host'
        elif self.host_distance_total <= host_virial_radius:
            self.is_satellite = True
            self.type = 'satellite'
        else:
            self.is_field = True
            self.type = 'field'

    def _assign_type_from_halt(self):
        '''
        assign
            self.is_primary
            self.is_satellite
            self.is_field
            self.host_tree_index -- pointer to the tree
            self.host_distance
            self.host_distance_total
        '''
        assert self.halt is not None

        if 'host.distance' not in self.halt:
            rockstar.io.IO._assign_host_to_tree(self.halt, 'star')
        if self.two_hosts and 'host2.distance' not in self.halt:
            rockstar.io.IO._assign_host_to_tree(
                self.halt, 'star', host_number=2)

        if self.halt_prop('host.distance.total') == 0:
            self.is_satellite = False
            self.is_primary = True
            self.is_field = False
            # self.host_index = self.index
            self.host_distance = np.array([0, 0, 0], dtype='f')
            self.host_distance_total = 0.0
            return
        if self.two_hosts:
            if self.halt_prop('host2.distance.total') == 0:
                self.is_satellite = False
                self.is_primary = True
                self.is_field = False
                # self.host_index = self.index
                self.host_distance = np.array([0, 0, 0], dtype='f')
                self.host_distance_total = 0.0
                return

        snapshot = self.halt_prop('snapshot')

        host_distances_at_snapshot = self.halt.prop('host.distance.total')[
            self.halt.prop('snapshot') == snapshot]
        host_index = np.where(host_distances_at_snapshot == 0)[0][0]

        self.host_distance = self.center_position - \
            self.halt.prop('position')[host_index]
        self.host_distance_total = np.linalg.norm(self.host_distance)
        if (self.host_distance_total <= self.halt.prop('radius')[host_index]):
            self.is_satellite = 'host1'
            # self.host_index = host_index
            self.is_primary = False
            self.is_field = False
            return

        if two_hosts:
            host2_distances_at_snapshot = self.halt.prop('host2.distance.total')[
                self.halt.prop('snapshot') == snapshot]
            host2_index = np.where(host2_distances_at_snapshot == 0)[0][0]

            self.host2_distance = self.center_position - \
                self.halt.prop('position')[host2_index]
            self.host2_distance_total = np.linalg.norm(self.host_distance)
            if (self.host2_distance_total <= self.halt.prop('radius')[host2_index]):
                self.is_satellite = 'host2'
                self.is_primary = False
                self.is_field = False
                self.host_index = host_index
                return
        else:
            self.host2_distance = np.array([np.inf, np.inf, np.inf])
            self.host2_distance_total = np.inf

        # if i get this far, then I'm not (either) primary host or a satellite of either
        self.is_satellite = False
        self.is_primary = False
        self.is_field = True

    def get_is_satellite(self):
        if self.is_satellite is None:
            self.assign_type()
        return self.is_satellite

    def get_is_primary(self):
        if self.is_primary is None:
            self.assign_type()
        return self.is_primary

    def get_archeological_burstiness_windows(self, dt1=0.1, dt2=0.01, at_birth=True):
        large_dt = max([dt1, dt2])
        small_dt = min([dt1, dt2])
        assert large_dt > small_dt  # can't be the same

        SFR = self.get_archeological_sfr(small_dt, at_birth=at_birth)
        Bins = ut.binning.BinClass([0, self.snapshot['time']], width=large_dt)
        Bins.Burstiness = np.zeros(Bins.mids.size)

        for ii, (tlow, thigh) in enumerate(zip(Bins.mins, Bins.maxs)):
            # include bins w/ the mids inside, cause that means they're more than half in (and won't be included twice)
            in_bin = (SFR.mids >= tlow) & (SFR.mids < thigh)

            # problem:  the following doesn't work because log 0 = -inf, and scatter there is meaningless
            # # want to do scatter in the log, so take the log of the SFR then find the scatter in that
            # log_sfr = ut.math.get_log(SFR.StarFormationRate[in_bin])
            # Bins.Burstiness[ii] = np.std(log_sfr)
            # so instead, let's do log of the scatter:
            Bins.Burstiness[ii] = ut.math.get_log(
                np.std(SFR.StarFormationRate[in_bin]))
        Bins.Burstiness[np.isnan(Bins.Burstiness)] = 0

        if self.burstiness is None:
            self.burstiness = Bins
        return Bins

    def get_archeological_sfr(self, dt=0.1, at_birth=True):
        formation_time = self.star_particle_prop('form.time')
        if at_birth:
            mass = self.star_particle_prop('form.mass')
        else:
            mass = self.star_particle_prop('mass')

        TimeBins = ut.binning.BinClass([0, self.snapshot['time']], width=dt)
        TimeBins.StellarMassFormed = TimeBins.get_histogram(
            formation_time, weights=mass)
        TimeBins.StarFormationRate = TimeBins.StellarMassFormed / TimeBins.widths

        if self.sfr is None:
            self.sfr = TimeBins
        return TimeBins

    def get_archeological_sfh(self, at_birth=True, zero=True):
        """
        get the *exact* archaelogical SFH 
        if at_birth is True, then use the birth masses of the stars 
            (i.e., rewind stellar mass loss).  otherwise uses the
            present masses of the stars
        """
        from .histograms import cumulative_histogram

        xprop = 'form.scalefactor'

        aform = self.star_particle_prop('form.scalefactor')
        tform = self.star_particle_prop('form.time')
        if at_birth:
            mass = self.star_particle_prop('form.mass')
        else:
            mass = self.star_particle_prop('mass')

        sfh, scale = cumulative_histogram(aform, weights=mass, zero=True)
        cosmic_time = np.unique(tform)
        if zero:
            cosmic_time = np.concatenate(([cosmic_time[0]-1e-10], cosmic_time))

        redshift = (1./scale) - 1
        lookback_time = self.snapshot['time'] - cosmic_time

        normalized_sfh = sfh / sfh[-1]

        res = {'scalefactor': scale, 'scale': scale,
               'cosmic.time': cosmic_time, 'time': cosmic_time,
               'lookback.time': lookback_time, 'time.lookback': lookback_time,
               'cumulative.sfh': sfh, 'normalized.cumulative.sfh': normalized_sfh}

        if self.sfh is None:
            self.sfh = res

        return res

    def get_sfh_moments(self, sfh=None, percentiles=[5, 10, 25, 50, 90, 95, 100], durations=[[10, 90]], masses=[1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]):
        """
        get different moments (i.e., important times) of the star formation history.
        returns a dictionary w/ entires 
            'time.'+percentiles[ii] 
                corresponds to the cosmic time when the SFH hit each percentile. 
            'duration.'+durations[ii][0]+'.to.'+durations[ii][1]
                corresponds to the time difference between when the SFH hit the first
                percentile and the second percentile
            'mass.time.'+masses[ii]
                corresponds to the cosmic time when the non-normalized SFH first reaches
                this mass
        """
        if sfh is None:
            if self.sfh is None:
                sfh = self.get_archeological_sfh()
            else:
                sfh = self.sfh
        res = {}
        for p in percentiles:
            idx = nearest_index(p/100., sfh['normalized.cumulative.sfh'])
            res['time.'+str(p)] = sfh['time'][idx]
        for (pmin, pmax) in durations:
            idx_min = nearest_index(
                pmin/100., sfh['normalized.cumulative.sfh'])
            idx_max = nearest_index(
                pmax/100., sfh['normalized.cumulative.sfh'])
            res['duration.'+str(pmin)+'.to.'+str(pmax)
                ] = np.abs(sfh['time'][idx_max] - sfh['time'][idx_min])
        for mass in masses:
            if mass > sfh['cumulative.sfh'].max():
                continue

            # first time that the galaxy had formed this much mass (different from when it first hit this mass)
            idx = nearest_index(mass, sfh['cumulative.sfh'])
            res['mass.time.{:.0e}'.format(mass)] = sfh['time'][idx]
        if self.sfh_moments is None:
            self.sfh_moments = res
        else:
            self.sfh_moments = {**self.sfh_moments, **res}
        return res

    def build_hals_main_branch_indices(self, do_host=False):
        """
        build/get the index in each hals[ii] of the main branch
        progenitor of this halo or of a host
        """

        def get_mmp_index(my_id, previous_hal, mmp_prop='vel.circ.max'):
            # first get the indices in the previous catalog where the descendant is my ID
            if not len(previous_hal):
                return -2**31

            progenitor_indices = np.where(
                previous_hal.prop('descendant.id') == my_id)[0]
            if not progenitor_indices.size:
                return -2**31

            # then get the sorting values of the progenitor halos
            progenitor_mmp_prop = previous_hal.prop(
                mmp_prop, progenitor_indices)

            # then return the index of the one that's the biggest
            return progenitor_indices[np.argmax(progenitor_mmp_prop)]

        assert self.hals is not None
        if do_host == True or do_host == 'host' or do_host == 'host1':
            starting_index = self.hals[-1].prop('host.index')[0]
            store_name = 'host_hals_mb_indices'
        elif do_host == 'host2' or do_host == '2':
            starting_index = self.hals[-1].prop('host2.index')[0]
            store_name = 'host2_hals_mb_indices'
        else:
            assert self.index is not None
            starting_index = self.index
            store_name = 'hals_mb_indices'

        res = np.empty(len(self.hals), dtype=int)
        res.fill(-2**31)

        current_snapshot_index = len(self.hals) - 1
        my_index = starting_index
        while my_index >= 0:
            res[current_snapshot_index] = my_index

            my_id = self.hals[current_snapshot_index].prop('id', my_index)
            my_index = get_mmp_index(
                my_id, self.hals[current_snapshot_index-1])
            current_snapshot_index -= 1

        self.__dict__[store_name] = res
        return res

    def get_main_branch_indices(self):
        """
        set/return the indices of the main branch of the halo in the halt tree

        really just a thin wrapper around self.halt.prop('progenitor.main.indices')
        """

        assert self.halt is not None
        prog_main_index = self.halt_index
        prog_main_indices = self.halt.prop(
            'progenitor.main.indices', self.halt_index)
        self.main_branch_indices = prog_main_indices
        return prog_main_indices

    def get_prop_evolution(self, prop_name='mass', preferred_method='halt', do_host=False):
        """
        get the evolutionary history of a given property.  

        ::preferred_method:: determines the preferred way of getting that evolution:
            preferred_method = 'halt' goes to the halo merger tree, halt, first (presumably created by ctrees)
            preferred_method = 'hals' goes to the raw halo catalogs from Rockstar using my internal tools to try to stitch together
            preferred_method = 'ytree' using the arbor

        if the preferred_method is not available, then we'll first try the halt, then the arbor, then the hals
        """
        if preferred_method == 'halt' and self.halt is not None:
            method = self._halt_prop_evolution
        elif preferred_method == 'hals' and self.hals is not None:
            method = self._hals_prop_evolution
        elif preferred_method == 'ytree' and self.arbor is not None:
            method = self._arbor_prop_evolution

        else:
            print(
                "! -- warning:  cannot compute evolution using {}".format(preferred_method))
            if self.halt is not None:
                print("! -- falling back to halt")
                method = self._halt_prop_evolution
            elif self.arbor is not None:
                print("! -- falling back to arbor")
                method = self._arbor_prop_evolution
            elif self.hals is not None:
                print("! -- falling back to hals")
                method = self._hals_prop_evolution
            else:
                raise IOError("No way to get time evolution")

        return method(prop_name, do_host=do_host)

    def get_position_track(self, xprop='scalefactor', preferred_method='halt', do_host=False):
        """
        same as get_prop_evolution, but returns an interpolatable function as well
        """
        from scipy.interpolate import interp1d
        X_of_t = self.get_prop_evolution(
            prop_name='position', preferred_method=preferred_method, do_host=do_host)
        pos = X_of_t['position']
        xvals = X_of_t[xprop]

        msk = xvals >= 0
        xvals = xvals[msk]
        pos = pos[msk]

        position_function = interp1d(xvals, pos, axis=0, kind='cubic')
        self.__dict__['position_of_'+xprop] = position_function
        return position_function

    def _halt_prop_evolution(self, prop_name, do_host=False):
        assert self.halt is not None

        # figure out if I'm grabbing track for host or gal
        if do_host == True or do_host == 'host' or do_host == 'host1':
            mb_indices = self.halt.prop(
                'progenitor.main.indices', self.halt.prop('host.index')[0])
            store_name = 'host_'+prop_name.replace('.', '_')+'_evolution'
        elif do_host == 'host2' or do_host == '2':
            mb_indices = self.halt.prop(
                'progenitor.main.indices', self.halt.prop('host2.index')[0])
            store_name = 'host2_'+prop_name.replace('.', '_')+'_evolution'
        else:
            if self.main_branch_indices is None:
                self.get_main_branch_indices()
            mb_indices = self.main_branch_indices
            store_name = prop_name.replace('.', '_')+'_evolution'

        result = {}

        values = self.halt.prop(prop_name, mb_indices)
        snaps = self.halt.prop('snapshot', mb_indices)

        # convert snapshots into times:
        times = self.halt.Snapshot['time'][snaps]
        scales = self.halt.Snapshot['scalefactor'][snaps]
        redshifts = self.halt.Snapshot['redshift'][snaps]

        # store sorted by increasing snapshot value
        sorti = np.argsort(snaps)
        result[prop_name] = values[sorti]
        result['snapshot'] = snaps[sorti]
        result['time'] = times[sorti]
        result['scalefactor'] = scales[sorti]
        result['redshift'] = redshifts[sorti]

        self.__dict__[store_name] = result
        return result

    def _hals_prop_evolution(self, prop_name, do_host=False):
        """
        get the evolutionary history of a given property from the tree in the hals 
        """
        assert self.hals is not None

        # figure out if I'm grabbing track for host or gal
        if do_host == True or do_host == 'host' or do_host == 'host1':
            mb_indices = self.build_hals_main_branch_indices(do_host='host')
            store_name = 'host_'+prop_name.replace('.', '_')+'_evolution'
        elif do_host == 'host2' or do_host == '2':
            mb_indices = self.build_hals_main_branch_indices(do_host='host2')
            store_name = 'host2_'+prop_name.replace('.', '_')+'_evolution'
        else:
            if self.hals_mb_indices is None:
                self.build_hals_main_branch_indices()
            mb_indices = self.hals_mb_indices
            store_name = prop_name.replace('.', '_')+'_evolution'

        result = {}
        fill_value = -2**31

        result['snapshot'] = np.empty(len(self.hals), dtype=int)
        dtype = self.hals[-1].prop(prop_name).dtype
        if self.hals[-1].prop(prop_name).ndim > 1:
            shape = (len(self.hals), self.hals[-1].prop(prop_name).shape[1])
            result[prop_name] = np.empty(shape, dtype=dtype)
        else:
            result[prop_name] = np.empty(len(self.hals), dtype=dtype)

        result['snapshot'].fill(fill_value)
        result[prop_name].fill(fill_value)

        for ii, hals_index in enumerate(self.hals_mb_indices):
            if hals_index < 0 or not len(self.hals[ii]):
                continue

            result[prop_name][ii] = self.hals[ii].prop(prop_name, hals_index)
            result['snapshot'][ii] = self.hals[ii].snapshot['index']

        have_halo = result['snapshot'] > 0

        # can use any of the halo catalogs to convert from snapshot to time etc.
        result['time'] = np.empty(result['snapshot'].size, dtype=float)
        result['scalefactor'] = np.empty(result['snapshot'].size, dtype=float)
        result['redshift'] = np.empty(result['snapshot'].size, dtype=float)

        result['time'].fill(fill_value)
        result['scalefactor'].fill(fill_value)
        result['redshift'].fill(fill_value)

        result['time'][have_halo] = self.hals[-1].Snapshot['time'][result['snapshot'][have_halo]]
        result['scalefactor'][have_halo] = self.hals[-1].Snapshot['scalefactor'][result['snapshot'][have_halo]]
        result['redshift'][have_halo] = self.hals[-1].Snapshot['redshift'][result['snapshot'][have_halo]]

        # now sort in increasing time -- fill values will go at beginning:
        sorti = np.argsort(result['snapshot'])
        for key in result:
            result[key] = result[key][sorti]

        # now trim off places where I don't have the halo:
        msk = result['time'] > 0
        for key in result:
            result[key] = result[key][msk]

        self.__dict__[store_name] = result
        return result

    def _arbor_prop_evolution(self, prop_name, do_host=False):
        assert self.arbor is not None

        # figure out if I'm grabbing track for host or gal
        if do_host == True or do_host == 'host' or do_host == 'host1':
            root_index = np.argsort(self.arbor['mass'])[-1]
            root_node = self.arbor[root_index]
            store_name = 'host_'+prop_name.replace('.', '_')+'_evolution'
        elif do_host == 'host2' or do_host == '2':
            root_index = np.argsort(self.arbor['mass'])[-2]
            root_node = self.arbor[root_index]
            store_name = 'host2_'+prop_name.replace('.', '_')+'_evolution'
        else:
            assert self.arbor_root is not None
            root_node = self.arbor_root
            store_name = prop_name.replace('.', '_')+'_evolution'

        result = {}

        if 'host' in prop_name:
            assert 'distance' in prop_name or 'velocity' in prop_name
            if 'host2' in prop_name:
                host_root_index = np.argsort(self.arbor['mass'])[-2]
            else:
                host_root_index = np.argsort(self.arbor['mass'])[-1]

            if 'distance' in prop_name:
                host_values = self.arbor[host_root_index]['prog', 'position']
                gal_values = self.arbor_root['prog', 'position']
            elif 'velocity' in prop_name:
                host_values = self.arbor[host_root_index]['prog', 'velocity']
                gal_values = self.arbor_root['prog', 'velocity']

            host_redshift = self.arbor[host_root_index]['prog', 'redshift']
            gal_redshift = root_node['prog', 'redshift']

            redshift, host_indices, gal_indices = np.intersect1d(
                host_redshift, gal_redshift, return_indices=True)
            host_values = host_values[host_indices]
            gal_values = gal_values[gal_indices]

            values = ut.coordinate.get_distances(
                gal_values, host_values, total_distance='total' in prop_name)
        else:
            values = root_node['prog', prop_name]
            redshift = root_node['prog', 'redshift']

        sorti = np.argsort(redshift)
        values = values[sorti].values
        redshift = redshift[sorti]

        result = {}
        result[prop_name] = values
        result['redshift'] = redshift

        # convert to time, scale factor, and snapshot
        result['snapshot'] = self.Snapshot.get_snapshot_indices(
            time_kind='redshift', values=redshift)
        result['time'] = self.Snapshot['time'][result['snapshot']]
        result['scalefactor'] = 1./(1+result['redshift'])
        self.__dict__[prop_name.replace('.', '_')+'_evolution'] = result
        return result

    def get_orbit_history(self, preferred_method='halt', wrt_host2=False, infall_radius=-1, interpolate=0):
        """
        gets + assigns 
            * distance from the host (or host2) as a function of scalefactor, time, and redshift
            * pericentric distance
            * xprop of pericentric distance
            * first infall time (defined as crossing halt.prop('radius'))
            * last infall time

        if infall_radius > 0, then use a fixed physical radius to define infall.
        otherwise, grabs halt.prop('radius') for the host at each time

        if interpolate is > 0, then distance will be interpolated w/ a cubic spline
        over `interpolate` points in linear space.  if interpolate <= number of points
        already in there, then there's no interpolation done
        """
        host = 'host'
        if wrt_host2:
            host = 'host2'

        dist_prop = host+'.distance.total'
        total_distance_evolution = self.get_prop_evolution(
            prop_name=dist_prop, preferred_method=preferred_method)

        distance = total_distance_evolution[dist_prop]
        time = total_distance_evolution['time']

        interpolated = False
        # may need this for the host properties later
        uninterpolated_time = np.array(time, copy=True)
        uninterpolated_snapshots = np.array(
            total_distance_evolution['snapshot'], copy=True)

        if interpolate >= time.size:
            from scipy.interpolate import interp1d
            from astropy.cosmology import z_at_value
            from astropy import units as u

            interpolated = True

            tmin, tmax = time.min(), time.max()
            test_xvals = np.linspace(xmin, xmax, interpolate)
            interpolated_distance_function = interp1d(
                time, distance, kind='cubic')

            distance = interpolated_distance_function(test_xvals)
            time = test_xvals

            total_distance_evolution[dist_prop] = distance
            total_distance_evolution['time'] = time

            total_distance_evolution['redshift'] = np.array(
                [z_at_value(self.Cosmo.age, tx*u.Gyr) for tx in test_xvals])
            total_distance_evolution['scalefactor'] = 1. / \
                (1+total_distance['redshift'])

            # only label the array values that correspond to actual snapshots
            interpolated_snapshots = np.ones(time.size) * -2**31
            sorter = np.argsort(uninterpolated_time)
            # find which indices in the interpolated times are closest to uninterpolated times
            indices = np.searchsorted(time, uninterpolated_time, sorter=sorter)
            interpolated_snapshots[indices] = uninterpolated_snapshots

            total_distance_evolution['snapshot'] = interpolated_snapshots

        # basically just want to add to the information already in total_distance_evolution
        result = total_distance_evolution

        # add some key times, abeginning with the easier one of pericenter:
        pericenter_index = np.argmin(distance)
        result['pericenter.index'] = pericenter_index
        result['pericenter.distance'] = distance[pericenter_index]
        result['pericenter.time'] = time[pericenter_index]
        result['pericenter.redshift'] = result['redshift'][pericenter_index]
        result['pericenter.scalefactor'] = result['scalefactor'][pericenter_index]

        # now do infall
        if infall_radius > 0:
            # easy case -- when did we cross some fixed radius?
            host_radius = infall_radius * np.ones(time.size)
            host_time = time
        else:
            # harder case -- when did we cross radius(t)?

            # need to compute the radius of the host I'm looking at at each time
            host_radius_evolution = self.get_prop_evolution(
                'radius', preferred_method=preferred_method, do_host=host)
            host_radius = host_radius_evolution['radius']
            host_time = host_radius_evolution['time']

            if interpolated:
                # need to interpolate the host radii too:
                host_radius_of_time = interp1d(
                    host_time, host_radii, kind='cubic')
                host_radius = host_radius_of_time(time)
                host_time = time

        # get the times that this galaxy and the host both exists at -- need to do whether I interpolate or not
        common_times, host_indices, gal_indices = np.intersect1d(
            host_time, time, return_indices=True)

        # trim both down to the snapshots at which both exist -- these should be at the same time now
        distance_for_comparison = distance[gal_indices]
        host_radius_for_comparison = host_radius[host_indices]

        distance_minus_radius = distance_for_comparison - \
            host_radius_for_comparison  # positive => outside, negative => inside

        # look for the times that the value goes from positive to negative -- this indexes common_times
        # need the + 1 cause otherwise I get the index before; I want the index after
        crossings = np.where(np.diff(np.sign(distance_minus_radius)))[0] + 1

        ncrossings = crossings.size
        if ncrossings == 0:
            result['first.infall.index'] = None
            result['first.infall.time'] = np.inf
            result['first.infall.redshift'] = -np.inf
            result['first.infall.scalefactor'] = np.inf

            result['last.infall.index'] = None
            result['last.infall.time'] = np.inf
            result['last.infall.redshift'] = -np.inf
            result['last.infall.scalefactor'] = np.inf
        else:
            # these can be the same; doesn't matter, as long as I have at least 1.  remember that these index snapshot_values_in_common
            if common_times[0] > common_times[-1]:
                first_infall_index = crossings[-1]
                last_infall_index = crossings[0]
            else:
                first_infall_index = crossings[0]
                last_infall_index = crossings[-1]

            if interpolated:
                # have to get the redshift from the time, since that's what I index
                result['first.infall.time'] = common_times[first_infall_index]
                result['first.infall.redshift'] = z_at_value(
                    self.Cosmo.age, result['first.infall.time']*u.Gyr)

                result['last.infall.time'] = common_times[last_infall_index]
                result['last.infall.redshift'] = z_at_value(
                    self.Cosmo.age, result['last.infall.time']*u.Gyr)
            else:
                common_redshifts = result['redshift'][gal_indices]

                result['first.infall.time'] = common_times[first_infall_index]
                result['first.infall.redshift'] = common_redshifts[first_infall_index]

                result['last.infall.time'] = common_times[last_infall_index]
                result['last.infall.redshift'] = common_redshifts[last_infall_index]

            # either way, calculate the scalefactor from the redshift
            result['first.infall.scalefactor'] = 1. / \
                (1 + result['first.infall.redshift'])
            result['last.infall.scalefactor'] = 1. / \
                (1 + result['last.infall.redshift'])

        self.__dict__[host+'_orbit_history'] = result
        return result

    def _parse_center_position(self, kwargs):
        # check if center_position was passed in in these kwargs in case I'm doing halo properties of something else
        return np.array(kwargs.get('center_position', self.center_position), copy=True)

    def _clean_kwargs(self, kwargs, valid_kwargs):
        all_kwargs = {**self.kwargs, **kwargs}
        return dict([(key, all_kwargs[key]) for key in all_kwargs if key in valid_kwargs])

    def get_galaxy_properties(self, **kwargs):
        assert self.part is not None
        center_position = self._parse_center_position(kwargs)

        valid_kwargs = ['species_name', 'distance_max', 'principal_axes_distance_max',
                        'edge_kind', 'edge_value', 'distance_bin_width', 'distance_scaling', 'axis_kind',
                        'principal_axes_vectors', 'other_axis_distance_limits']
        # this_kwargs = dict([(key, kwargs[key]) for key in {**self.kwargs, **kwargs} if key in valid_kwargs])
        this_kwargs = self._clean_kwargs(kwargs, valid_kwargs)

        return ut.particle.get_galaxy_properties(self.part, center_position=center_position, **this_kwargs)

    def get_halo_properties(self, **kwargs):
        assert self.part is not None
        if 'dark' not in self.part:
            raise Warning(
                "!! -- asking for halo properties, but didn't pass in dark matter particles!  expect error!")

        center_position = self._parse_center_position(kwargs)
        valid_kwargs = ['species', 'virial_kind', 'distance_min', 'distance_max',
                        'distance_bin_width', 'distance_scaling']
        # this_kwargs = dict([(key, kwargs[key]) for key in {**self.kwargs, **kwargs} if key in valid_kwargs])
        this_kwargs = self._clean_kwargs(kwargs, valid_kwargs)

        return ut.particle.get_halo_properties(self.part, center_position=center_position, **this_kwargs)

    def image_yt(self, outname, species='star', width=20, indices_only=False, **kwargs):
        assert self.part is not None
        from sgklibs.yt_projection import yt_projection
        pos = self.part[species].prop('position')
        mass = self.part[species].prop('mass')

        if indices_only and species == 'star':
            pos = pos[self.star_indices]
            mass = mass[self.star_indices]

        center_position = self._parse_center_position(kwargs)
        prj = yt_projection(pos, mass, outname, bsize=width,
                            center=center_position, **kwargs)
        if self.prj is None:
            self.prj = prj
        return prj

    def image_raytrace(self, outname, particle_file=None, **kwargs):
        if particle_file is None:
            assert self.particle_file is not None
            particle_file = self.particle_file

        from visualization.image_maker import image_maker

        def snapfile_to_readargs(snapfile):
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

        snapdir, snapnum, snapshot_name = snapfile_to_readargs(particle_file)

        center_position = self._parse_center_position(kwargs)
        image_maker(snapnum, sdir=snapdir, snapshot_subdirectory='',
                    center=center_position, filename_set_manually=outname, **kwargs)

    def image_projected_density(self, ):
        # from viz_scripts.visualizations import #viz_part_2d, viz_part_indices, viz_with_halo
        raise NotImplementedError

    # TODO
    # add hoooks into computing profiles and stuff like that, cause why not now that I have part and center in one place
    def compute_radial_profile(self, ):
        raise NotImplementedError
