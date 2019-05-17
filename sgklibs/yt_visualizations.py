from .low_level_utils import backup_file, isint, isfloat, boxmsk
from .particle_io import read_part_wetzel
import utilities as ut
import numpy as np
import matplotlib
matplotlib.use('agg', warn=False)

import yt

class ProjectParticlesYT():
    """
    make density projections of particle data using yt

    includes a convenient wrapper function for visualizing the outputs of 
    GIZMO simulations, though you must have gizmo_analysis installed.

    supports both yt3 and yt4 (though it has to handle them in different ways).
    handles projections along any axis and also allows the figure to be rotated
    in the plane of the image by n*pi/2.  optionally has support for adding 
    circles around the center, around halos in halo catalogs,  points at 
    specified (x,y,z) positions, labels, and colorbar stuff.  
    
    Attributes:
        proj (:ob:`yt.visualization.plot_window.ProjectionPlot`):  A 
            yt projection plot, though this only exists once either 
            `yt3x_projection` or `yt4x_projection` is called.  
        dpi (int):  The dots-per-inch used in saving the figure.  
            will be set automatically to the required number to have each
            buffer element >= 1 pixel in size, but starts off at 300.
        self.figure_size (float):  size of the image, in inches, when saved.
            also scales the relative size of any text, which is scaled by 
            the fontsize in points.
        fontsize (float):  font size to use for labels, etc.
        width_mult_buffer (float):  overflow area around the box
            to allow when trimming particles before passing them to 
            yt.  should be > 1 -- setting = 1 can give edge effects
        yt4 (bool):  Whether or not the imported version of yt parses
            to be >= 4.
            
    """

    def __init__(self):
        from packaging import version
        if version.parse(yt.__version__) > version.parse('3.9'):
            self.yt4 = True
        else:
            self.yt4 = False

        self.figure_size = 10
        self.fontsize = 28
        self.dpi = 300
        self.proj = None
        self.width_mult_buffer = 1.125


    def get_bbox(self, width):
        """
        return a valid bounding box given a side length of a cube 
        
        Args:
            width (float):  total side length of the cube
        
        Returns:
            bbox:  a 3x2 array that bounds the total width of 
                the cube plus a given by self.width_mult_buffer.
        """
        bl = [-width*self.width_mult_buffer/2, width*self.width_mult_buffer/2]
        return np.array([bl, bl, bl])


    def prepare_data_dictionary(self, cut_positions, cut_weights, ptype, axis, img_rotate):
        """
        create a dictionary to pass to yt from arrays of particle data
        
        particle data (positions and weights) should have already been 
        trimmed before handing to this function.
        
        Args:
            cut_positions (array-like):  Nx3 array of (x,y,z) particle 
                positions, shifted such that the center of the cube to 
                be visualized is at (0, 0, 0) and no particles lie 
                outside the bounding box (+ buffer) you plan to visualize.
            
            cut_weights (array-like):  len-N array giving the weights 
                or masses of the particles
            
            ptype (str):  name of the particle species, for purposes of
                getting the colorbar correct.
           
            axis (int or str):  either 'x', 'y', 'z', 1, 2, or 3 to give 
                the axis you want to project the particles down (i.e., 
                'x' will give the density in the y-z plane).
            img_rotate (int):  either 0, 90, 180, or 270 to rotate the 
                image (in the plane of the image) by that many degrees
        
        Returns:
            data (dict):  a dictionary with keys (ptype, 'particle_X'),
                where X is one of 'position_x', 'position_y', 'position_z',
                and 'mass', with x-y-z (and inverted etc.) set such that 
                the projection along and rotation about the axis will be 
                correct.
        """
        indices, inversions = self.parse_img_rotate(img_rotate, axis)

        data = {}
        data[(ptype, 'particle_mass')] = cut_weights
        for ii, ax in enumerate(['x', 'y', 'z']):
            data[(ptype, 'particle_position_'+ax)] = cut_positions[:, indices[ii]]*inversions[ii]

        return data


    def save_proj(self, outname):
        """
        save the projection with the proper dpi
        
        Args:
            outname (str):  path to save the image
        """
        assert self.proj is not None, "cannot save the projection before it is created"
        self.proj.save(outname, mpl_kwargs=dict(dpi=self.dpi))

    def yt3x_projection(self, cut_positions, cut_weights, width, 
        n_ref=8, axis=0, method='mip', pixels=2048, img_rotate=0, 
        length_unit='kpc', mass_unit='Msun', interpolation='bicubic', 
        **kwargs):
        """
        use the yt-3 method of creating a ProjectionPlot out of 
        particle data defined cut_positions and cut_weights, 

        Args:
            * cut_positions (N x 3 array):  array of particle positions

            * cut_weights (len N array):  array of particle weights

            * width (float):  size of box to visualize, in length_unit

            * n_ref (int):  number of particles to allow in a cell before
                            refining (i.e. refines cells with > n_ref particles).
                            smaller values give a higher resolution image

            * axis (int or str):  axis to project along

            * method (str):  method to pass to ProjectionPlot.  most 
                             likely either 'integrate' for LOS densities
                             or 'mip' for max-in-pixel volume densities

            * pixels (int):  number of pixels in the image 
                             (scales the dpi; uses a fixed image size)

            * img_rotate (int):  either 0, 90, 180, or 270 to rotate the image 
                                 (in the plane) through that many degrees. 

            * length_unit (str):  unit of the cut_positions

            * mass_unit (str):  unit of the cut_weights

            * interpolation (str):  type of interpolation to apply to the image

            ** kwargs:  passed to yt.ProjectionPlot


        Returns:
            a yt.ProjectionPlot instance
        """

        data = self.prepare_data_dictionary(cut_positions, cut_weights, 'io', axis, img_rotate)
        bbox = self.get_bbox(width)

        ds = yt.load_particles(data, length_unit=length_unit, mass_unit=mass_unit, bbox=bbox, 
            n_ref=n_ref, periodicity=(False, False, False))

        self.field = ('deposit', 'io_density')
        self.field_name = self.field[1]

        self.proj = yt.ProjectionPlot(ds, axis, self.field, method=method, 
            center=np.zeros(3), width=(width, length_unit), **kwargs)

        plot = self.proj.plots[list(self.proj.plots)[0]]
        ax = plot.axes
        img = ax.images[0]
        img.set_interpolation(interpolation)

        self.proj.set_buff_size((pixels, pixels))
        self.proj.set_figure_size(self.figure_size)

        ## store the appropriate DPI for later saving
        self.dpi = pixels / self.figure_size 

        self.set_bgcolor_and_units(method)

        return self.proj

    def yt4x_projection(self, cut_positions, cut_weights, width, 
        axis=0, method='mip', pixels=2048, img_rotate=0, 
        length_unit='kpc', mass_unit='Msun', **kwargs):
        """
        use the yt-4 method of creating a ProjectionPlot out of 
        particle data defined cut_positions and cut_weights, 

        Args:
            * cut_positions (N x 3 array):  array of particle positions

            * cut_weights (len N array):  array of particle weights

            * width (float):  size of box to visualize, in length_unit

            * axis (int or str):  axis to project along

            * method (str):  method to pass to ProjectionPlot.  most 
                             likely either 'integrate' for LOS densities
                             or 'mip' for max-in-pixel volume densities

            * pixels (int):  number of pixels in the image 
                             (scales the dpi; uses a fixed image size)

            * img_rotate (int):  either 0, 90, 180, or 270 to rotate the image 
                                 (in the plane) through that many degrees. 

            * length_unit (str):  unit of the cut_positions

            * mass_unit (str):  unit of the cut_weights

            * ptype (str):  particle type being visualized.  only matters for the colorbar.

            ** kwargs:  passed to yt.ProjectionPlot


        Returns:
            a yt.ProjectionPlot instance
        """
        import yt

        data = self.prepare_data_dictionary(cut_positions, cut_weights, 'io', axis, img_rotate)
        bbox = self.get_bbox(width)

        ds = yt.load_particles(data, length_unit=length_unit, mass_unit=mass_unit, bbox=bbox,
            periodicity=(False, False, False))

        ds.add_sph_fields()

        self.field = ('io', 'density')
        self.field_name = self.field[1]

        self.proj = yt.ProjectionPlot(ds, axis, self.field, center=np.zeros(3), 
            width=(width, length_unit), buff_size=(pixels, pixels), method=method)

        self.set_bgcolor_and_units(method)

        self.proj.set_figure_size(self.figure_size)
        self.dpi = pixels / self.figure_size

        return self.proj


    def add_particle_points(self, cut_points_to_plot, img_rotate, axis, color):
        """
        add individual dots to the image, e.g. to represent another particle species
        
        Args:
            cut_points_to_plot (array):  N x 3 array giving the x-y-z positions 
                (in the same frame as the original data, but again shifted such 
                that the visualization center is at (0, 0, 0)) to add dots to the 
                image
            img_rotate (int):  either 0, 90, 180, or 270 to rotate the 
                image (in the plane of the image) by that many degrees
            axis (int or str):  axis to project along
            color (str):  valid matplotlib color to apply to the points                        
        """
        
        data = self.prepare_data_dictionary(cut_points_to_plot, np.zeros(cut_points_to_plot.shape[0]), 'points', axis, img_rotate)

        print("adding markers for {} points".format(data[('points', 'particle_position_x')].size))
        my_points = [data[('points', 'particle_position_'+ax)] for ax in ['x', 'y', 'z']]

        return self.proj.annotate_marker(my_points, marker=',', coord_system='data', 
            plot_args={'color': color, 's': (72./self.dpi)**2})  # trying to make the size be a single pixel...


    def circle_center(self, radius, circle_args={}):
        """
        add a circle about the center of the image, e.g. to indicate a halo size
        
        args:
            radius (float):  size of the circle (in the same units
                as the particle data) to put on the projection.
            circle_args (dict):  dictionary of arguments passed
                to the matplotlib circle object to style it.
        """
        
        self.proj = self.proj.annotate_sphere(center=[0, 0, 0], 
            radius=radius, circle_args=circle_args)
        return self.proj


    def set_bgcolor_and_units(self, method, unit=None):
        """
        set the background color and units of the colorbar

        should be run on every plot
        
        Args:
            method (str):  the method used to create the plot, for knowing
                whether we're doing a 2D or 3D value.
            unit (str):  unit to use, if setting by hand.
        """
        assert self.proj is not None, "must be run after the projection is created"
        self.proj.set_background_color(self.field_name)

        if unit is not None:
            self.proj.set_unit(self.field_name, unit)
        elif method == 'mip':
            self.proj.set_unit(self.field_name, 'Msun/pc**3')
        else:
            self.proj.set_unit(self.field_name, 'Msun/pc**2')
        return self.proj


    def clean_up_proj(self, cmap=None, colorbar_label=None, 
        vmax=None, vmin=None, dynamic_range=None, hide_cbar=False, 
        hide_axes=False, scalebar=None):
        """
        handle all the nicening of the projection


        Args:
            * cmap (str):  colormap to apply to the image.  leave at 
                           None to use the default

            * colorbar_label (str):  label to use for the colorbar (if
                there's a colorbar shown).  if None, leaves at default

            * vmax, vmin, dynamic_range (float):  top, bottom, and range
                                                  to use in the colorbar.
                                                  can provide any of vmax+vmin,
                                                  vmax+dynamic_range, vmin+dynamic_range,
                                                  or just dynamic_range (uses default vmax)
                                                  cannot provide all three.

            * hide_cbar (bool):  whether or not to hide the colorbar

            * hide_axes (bool):  whether or not to hide the axes

            * scalebar (None or float):  if None, sets the scale bar to a 
                                         reasonable length based on the width.

                                         if <0, doesn't show a scale bar.

                                         if >0, shows a scale bar of that length in kpc.
        """
        if vmin is not None and vmax is not None and dynamic_range is not None:
            raise ValueError("can only specify two out of three of vmin, vmax, and dynamic_range")

        assert self.proj is not None, "cannot clean up a projection before it exists"

        ## set the colormap, if requested
        if cmap is not None:
            self.proj.set_cmap(self.field_name, cmap)

        ### handle the limits of the colorbar (if asked to)
        if (vmin is not None) and (vmax is not None):
            self.proj.set_zlim(self.field_name, vmin, vmax)

        elif vmin is not None and dynamic_range is not None:
            self.proj.set_zlim(self.field_name, zmin=vmin,
                               zmax=None, dynamic_range=dynamic_range)
        
        elif vmax is not None and dynamic_range is not None:
            self.proj.set_zlim(self.field_name, zmin=None,
                               zmax=vmax, dynamic_range=dynamic_range)
        
        elif dynamic_range is not None:
            zmax = self.proj.plots[self.field].image.get_clim()[1]
            zmin = zmax / dynamic_range
            self.proj.set_zlim(self.field_name, zmin, zmax)


        ## set the label of the colorbar (if there is a colorbar)
        if hide_cbar:
            self.proj.hide_colorbar()
        elif colorbar_label is not None:
            self.proj.set_colorbar_label(self.field_name, colorbar_label)

        ## remove the axes if requested, and add a scalebar instead
        if hide_axes:
            self.proj.hide_axes(draw_frame=True)
            if scalebar is None:
                minfrac = 0.25
                maxfrac = 0.35
                pos = (minfrac/2.0 + 0.075, 0.05)
                self.proj.annotate_scale(min_frac=minfrac, max_frac=maxfrac, 
                    pos=pos, text_args=dict(size=self.fontsize))
            elif scalebar > 0:
                yloc = 0.05
                w = self.proj.width[0].item()
                xloc = ((scalebar/w)/2.0 + 0.075)
                pos = (xloc, yloc)
                prj = prj.annotate_scale(pos=pos, coeff=scalebar, unit='kpc', 
                    text_args=dict(size=self.fontsize))
            else:
                print("hiding all scalebars on a plot of width {}".format(self.proj.width[0].item()))

        return self.proj


    def recenter_and_trim_arrays(self, pos, center_position, width, weight=None):
        """
        trim down an array of particles, optionally handling the masses
        
        Args:
            pos (array):  input N x 3 array giving the (x, y, z) positions of
                a set of particles to select from.
            center_position (array):  length 3 array to center the particles on
            width (float):  total width of the box to select particles form
            weight (array or None):  either an input length N array giving 
                the masses or weights of the particles, or None
        
        Returns:
            cut_positions:  An N x 3 array of positions, shifted such that
                center_position is at (0, 0, 0), and trimmed such that none
                exist outside the box-size + buffer.
            cut_weights:  a length N array giving the values of the weight 
                array for the particles in the box, or None if weight is None.    
        """
        
        msk = boxmsk(pos, center_position, width*self.width_mult_buffer/2)
        cut_positions = pos[msk] - center_position
        if weight is not None:
            cut_weights = weight[msk]
        else:
            cut_weights = None
        
        return cut_positions, cut_weights
    
    
    def recenter_and_trim_part(self, part, center_position, spec, width, weight=None):
        """
        trim down a set of particles read in with gizmo_analysis.gizmo_io
        
        a wrapper around `recenter_and_trim_arrays`
        
        Args:
            part (dict-like):  A dictionary-like object that contains 
                particles separated by species and at least a 'position' 
                array within the specified species's dictionary.  
            center_position (array):  length 3 array giving the center
                of the visualization, in the same units as the positions.
            spec (str):  species of particles to grab
            width (float):  full-width of the box to grab particles within
            weight (str):  name of the field within part[spec] to use as 
                weights for the particles (or None to not do weights).

        Returns:
            cut_positions:  An N x 3 array of positions, shifted such that
                center_position is at (0, 0, 0), and trimmed such that none
                exist outside the box-size + buffer.
            cut_weights:  a length N array giving the values of the weight 
                field for the particles in the box, or None if weight is None.
        """
        
        ## allow a small buffer around the edge to avoid edge effects
        pos = part[spec]['position']
        if weight is not None:
            weight = part[spec][weight]
            
        return recenter_and_trim_arrays(pos, center_position, width, weight)


    def add_label(self, text, text_loc=(0.025, 0.975), coord_system='axis', 
        fontsize=None, text_args=dict(color='white', ha='left', va='top'),
        inset_box_args=dict(facecolor='None', edgecolor='None'), **kwargs):
        """
        add text to the projection, defaulting to the top left
        
        Args:
            text (str):  text to add to the figure
            text_loc (tuple):  length 2 tuple giving the position of the text
            coord_system (str):  coord_system to apply to the text_loc
            fontsize (float):  size of the text in points, or defaults to self.fontsize
            text_args (dict):  dictionary to pass to `matplotlib.pyplot.text` to 
                style the text
            inset_box_args (dict):  dictionary to pass to `matplotlib.pyplot.text` 
                to style the box put behind the text.
            kwargs:  passed to `self.proj.annotate_text`
        
        """
        assert self.proj is not None, "cannot add text before making a projection"

        if fontsize is None:
            text_args['fontsize'] = self.fontsize
        else:
            text_args['fontsize'] = fontsize

        self.proj.annotate_text(text_loc, text, coord_system=coord_system, 
            text_args=text_args, inset_box_args=inset_box_args, **kwargs)

        return self.proj


    def project_gizmo_snap(self, width, species=['star'], weight='mass', 
        center_position=None, host_index=0, outname_base=None, read_kwargs={}, 
        cmaps=None, **kwargs):
        """
        visualize a gizmo snapshot using yt.ProjectionPlot.  

        Args:
            * width (float):  width of the image to visualize

            * species (str or list):  species to visualize

            * center_position (None or len 3):  either None to find the 
                                                center automatically, or 
                                                a length-3 array giving the
                                                center position

            * host_index (int): index of the host (starting at zero) to 
                                center on if center_position is None

            * outname_base (None or str):  beginning of the path to save 
                                           the projections to.  adds the
                                           species to the end of each name.
                                           if None, then the projections aren't 
                                           saved

            * read_kwargs:  a dictionary of keyword arguments to pass to 
                            `gizmo_analysis.gizmo_io.Read.read_snapshots`,
                            e.g. simulation_directory or snapshot_directory

            * cmaps (str, None, or list):  if None, then uses the default 
                                           colormap for all species.  if a 
                                           list, then needs to be the same 
                                           length as `species`, and each 
                                           entry should be either a valid 
                                           colormap or None (in which case
                                           that species uses the default)

            ** kwargs:  any valid keyword arguments to `self.clean_up_proj`
                        or to either `self.yt4x_projection` or `self.yt3x_projection`,
                        whichever is appropriate 

        Returns:
            either a single projection (if only one species) or a list of 
            projections (one for each species)
        """
        from gizmo_analysis import gizmo_io

        if np.isscalar(species):
            species = [species]

        if cmaps is None:
            cmaps = [None] * len(species)

        if np.isscalar(cmaps):
            cmaps = [cmaps]
        assert len(cmaps) == len(species), "must provide a colormap for each species, or none at all"

        assign_host_coordinates = False
        host_number = 1
        if center_position is None:
            assign_host_coordinates = True
            host_number = host_index + 1
        
        part = gizmo_io.Read.read_snapshots(species=species, properties=['position', weight], 
            assign_host_coordinates=assign_host_coordinates, host_number=host_number, **read_kwargs)

        if center_position is None:
            center_position = part.host_positions[host_index]

        valid_cleanup_kwargs = self.clean_up_proj.__code__.co_varnames
        clean_up_kwargs = dict([(k, kwargs.pop(k)) for k in valid_cleanup_kwargs if k in kwargs])

        projections = []
        for ii, spec in enumerate(species):
            ## get the valid particle positions for this species and recenter
            cut_positions, cut_weights = self.recenter_and_trim_part(part, 
                center_position, spec, width, weight=weight)

            print("projecting {:,} {} particles, weighted by {}".format(cut_positions.shape[0], spec, weight))
            
            if self.yt4:
                self.yt4x_projection(cut_positions, cut_weights, width, **kwargs)
            else:
                self.yt3x_projection(cut_positions, cut_weights, width, **kwargs)

            ## clean up the axes as requested
            self.clean_up_proj(cmap=cmaps[ii], **clean_up_kwargs)

            ## save this projection and move on
            projections.append(self.proj)
            if outname_base is not None:
                self.save_proj(outname_base+'_'+spec+'.png')

        if len(projections) == 1:
            return projections[0]
        else:
            return projections


    def add_halos(self, hal, center_position=np.array([0, 0, 0]),
                  position_prop='position',
                  hal_indices=None, hal_cuts={},
                  radiusprop='radius',
                  circle_args={'facecolor': None,
                               'edgecolor': 'white', 'linewidth': 1.5},
                  img_rotate=0, axis=0,
                  ):
        '''
        adds circles around halos in a halo catalog to a projection plot

        Args:
            hal (`rockstar_analysis.rockstar_io.HaloDictionaryClass`):  a dictionary-like
                object, created by `rockstar_analysis.rockstar_io`, that contains arrays
                listing the properties of halos to add circles for.
            center_position (array):  length 3 array giving the center of the visualization
            position_prop (str):  name of the property to query `hal` for to get the centers
                of the halos
            hal_indices (array):  list of indices in `hal` of halos to circle
            hal_cuts (dict):  dictionary of 'key':[lower_limit, upper_limit] items that
                constrain the halos that are circled (e.g. hal_cuts={'vel.circ.max':[5, 150]}
                would only circle halos with 5 <= Vmax < 150).
            radiusprop (str):  property to query to get the radius of the circles to draw
            circle_args (dict):  dictionary of keyword arguments to pass to the matplotlib
                circle artist to style the circles.
            img_rotate (int):  angle that we've rotated the image in the plane 
                (0, 90, 180, and 270 only)
            axis (int or str):  axis that we've projected down
        '''
        import rockstar_analysis

        hal_positions = hal.prop(position_prop) - center_position
        indices, inversions = self.parse_img_rotate(img_rotate, axis)

        if hal_indices is None:
            if 'manual.mask' in hal:
                hal_cuts['manual.mask'] = [0.5, 1.5]  # add this one by hand
            if len(hal_cuts):
                hal_indices = ut.catalog.get_indices_catalog(hal, hal_cuts)
            else:
                hal_indices = np.arange(hal[radiusprop].size)

        print("Adding circles for {} halos in the catalog".format(hal_indices.size))
        for idx in hal_indices:
            c = np.array([hal_positions[idx][indices[ii]] *
                          inversions[ii] for ii in range(3)])
            self.proj.annotate_sphere(c, radius=(
                hal[radiusprop][idx], 'kpc'), circle_args=circle_args)
        return self.proj

    def parse_img_rotate(self, rotation, axis):
        """
        get indices and multipliers from an image rotation and axis to project along
        
        Args:
            rotation (int):  either 0, 90, 180, or 270 to rotate the image in the plane
                that many degrees
            axis (str or ing):  the axis to project along (either 'x' or 0 for x, 
                'y' or 1 for y, and 'z' or 2 for z)

        Returns:
            indices:  length-3 list of indices to apply to an x-y-z array to re-assign
                axes such that we project down the desired axis
            inversions:  length-3 list of 1 or -1 to multiply each axis by to perform 
                the appropriate rotation
                
            also sets both of these as attributes of the class
        """
        
        assert axis in [0, 1, 2, 'x', 'y', 'z']
        rename = {'x': 0, 'y': 1, 'z': 2}
        if axis in rename:
            axis = rename[axis]

        assert rotation in [0, 90, 180, 270]

        indices = None
        inversions = None
        # easy case first -- no rotation => no inversions
        if rotation == 0:
            indices = [0, 1, 2]
            inversions = [1.0, 1.0, 1.0]
        elif axis == 0:  # projecting along x axis => z and y are plotted by default, with y on the x-axis and z on the y-axis
            if rotation == 90:  # then I want -z on the y axis and +y on the x-axis
                indices = [0, 2, 1]
                inversions = [1.0, 1.0, -1.0]
            elif rotation == 180:  # effectively flip both axes
                indices = [0, 1, 2]
                inversions = [1.0, -1.0, -1.0]
            elif rotation == 270:
                indices = [0, 2, 1]
                inversions = [1.0, -1.0, 1.0]
        elif axis == 1:  # projecting along y axis => z and x are plotted by default, with x on the y-axis and z on the y-axis
            if rotation == 90:  # then I want -z on the y axis and +x on the x axis => z -> -x, x -> z
                indices = [2, 1, 0]
                inversions = [-1.0, 1.0, 1.0]
            elif rotation == 180:
                indices = [0, 1, 2]
                inversions = [-1.0, 1.0, -1.0]
            elif rotation == 270:
                indices = [2, 1, 0]
                inversions = [1.0, 1.0, -1.0]
        elif axis == 2:  # prjecting along z axis => x on x-axis, y on y-axis
            if rotation == 90:
                indices = [1, 0, 2]
                inversions = [1.0, -1.0, 1.0]
            elif rotation == 180:
                indices = [0, 1, 2]
                inversions = [-1.0, -1.0, 1.0]
            elif rotation == 270:
                indices = [1, 0, 2]
                inversions = [-1.0, 1.0, 1.0]
        
        if indices is None or inversions is None:
            KeyError(f"Passed in an invalid rotation ({rotation}) or axis ({axis})")
        
        self.indices = indices
        self.inversions = inversions
        return indices, inversions
            
YTProjection = ProjectParticlesYT()
