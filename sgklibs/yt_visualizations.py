from .low_level_utils import backup_file, isint, isfloat, boxmsk
from .particle_io import read_part_wetzel
import utilities as ut
import numpy as np
import matplotlib
matplotlib.use('agg', warn=False)

import yt

class ProjectParticlesYT():
    """
    a class to make density projections of particle data using yt

    supports both yt3.x and yt4.x (though has to handle them in different ways).
    optionally has support for adding circles for halos as well.  
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
        bl = [-width*self.width_mult_buffer/2, width*self.width_mult_buffer/2]
        return np.array([bl, bl, bl])


    def prepare_data_dictionary(self, cut_positions, cut_weights, ptype, axis, img_rotate):
        indices, inversions = self.parse_img_rotate(img_rotate, axis)

        data = {}
        data[(ptype, 'particle_mass')] = cut_weights
        for ii, ax in enumerate(['x', 'y', 'z']):
            data[(ptype, 'particle_position_'+ax)] = cut_positions[:, indices[ii]]*inversions[ii]

        return data


    def save_proj(self, outname):
        assert self.proj is not None, "cannot save the projection before it is created"
        self.proj.save(outname, mpl_kwargs=dict(dpi=self.dpi))

    def yt3x_projection(self, cut_positions, cut_weights, width, 
        n_ref=8, axis=0, method='mip', pixels=2048, img_rotate=0, 
        length_unit='kpc', mass_unit='Msun', **kwargs):
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
        data = self.prepare_data_dictionary(cut_points_to_plot, np.zeros(cut_points_to_plot.shape[0]), 'points', axis, img_rotate)

        print("adding markers for {} points".format(data[('points', 'particle_position_x')].size))
        my_points = [data[('points', 'particle_position_'+ax)] for ax in ['x', 'y', 'z']]

        return self.proj.annotate_marker(my_points, marker=',', coord_system='data', 
            plot_args={'color': color, 's': (72./self.dpi)**2})  # trying to make the size be a single pixel...


    def circle_center(self, radius, circle_args={}):
        self.proj = self.proj.annotate_sphere(center=[0, 0, 0], 
            radius=radius, circle_args=circle_args)
        return self.proj


    def set_bgcolor_and_units(self, method, unit=None):
        """
        set the background color and units of the colorbar

        should be run on every plot
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


    def recenter_and_trim_part(self, part, center_position, spec, width, weight=None):
        ## allow a small buffer around the edge to avoid edge effects
        msk = boxmsk(part[spec]['position'], center_position, width*self.width_mult_buffer/2)
        cut_positions = part[spec]['position'][msk] - center_position

        if weight is not None:
            cut_weights = part[spec][weight][msk]
        else:
            cut_weights = None

        return cut_positions, cut_weights


    def add_label(self, text, text_loc=(0.025, 0.975), coord_system='axis', 
        fontsize=None, text_args=dict(color='white', ha='left', va='top'),
        inset_box_args=dict(facecolor='None', edgecolor='None'), **kwargs):
        """
        add text to the projection, defaulting to the top left
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
                        or `yt.ProjectionPlot`

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


    def add_halos(self, hal, center=np.array([0, 0, 0]),
                  position_prop='position',
                  hal_indices=None, hal_cuts={},
                  radiusprop='radius',
                  circle_args={'facecolor': None,
                               'edgecolor': 'white', 'linewidth': 1.5},
                  img_rotate=0, axis=0,
                  ):
        '''
        given a dictionary of halos hal (read in via the rockstar_analysis
        package), adds them to an existing projection prj as circles 
        '''
        import rockstar_analysis

        hal_positions = hal.prop(position_prop) - center
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
        assert axis in [0, 1, 2, 'x', 'y', 'z']
        rename = {'x': 0, 'y': 1, 'z': 2}
        if axis in rename:
            axis = rename[axis]

        assert rotation in [0, 90, 180, 270]

        # easy case first -- no rotation, no inversions
        if rotation == 0:
            return [0, 1, 2], [1.0, 1.0, 1.0]

        if axis == 0:  # projecting along x axis => z and y are plotted by default, with y on the x-axis and z on the y-axis
            if rotation == 90:  # then I want -z on the y axis and +y on the x-axis
                return [0, 2, 1], [1.0, 1.0, -1.0]
            elif rotation == 180:  # effectively flip both axes
                return [0, 1, 2], [1.0, -1.0, -1.0]
            elif rotation == 270:
                return [0, 2, 1], [1.0, -1.0, 1.0]
        elif axis == 1:  # projecting along y axis => z and x are plotted by default, with x on the y-axis and z on the y-axis
            if rotation == 90:  # then I want -z on the y axis and +x on the x axis => z -> -x, x -> z
                return [2, 1, 0], [-1.0, 1.0, 1.0]
            elif rotation == 180:
                return [0, 1, 2], [-1.0, 1.0, -1.0]
            elif rotation == 270:
                return [2, 1, 0], [1.0, 1.0, -1.0]
        elif axis == 2:  # prjecting along z axis => x on x-axis, y on y-axis
            if rotation == 90:
                return [1, 0, 2], [1.0, -1.0, 1.0]
            elif rotation == 180:
                return [0, 1, 2], [-1.0, -1.0, 1.0]
            elif rotation == 270:
                return [1, 0, 2], [-1.0, 1.0, 1.0]
        raise KeyError("Passed in an invalid rotation or axis")


    ## ARCHIVE:  before modularizing the code
    # def __init__(self, positions, masses,
    #              outname=None,
    #              bsize=800, nref=8,
    #              img_rotate=0, axis=0,
    #              ptype='dark', center=np.array([0, 0, 0]),
    #              cmap='cubehelix',
    #              nocbar=True, noaxes=True,
    #              halos_to_add=None, haloargs={},
    #              fontsize=28, interpolation='bicubic',
    #              scalebar=None,
    #              vmin=None, vmax=None, dynamic_range=None,
    #              circle_radius=None, circle_args={},
    #              points_to_plot=None, points_color='red',
    #              number_density=False,
    #              dpi=300, figure_size=10,
    #              text_to_annotate=None,
    #              text_loc=(0.025, 0.975),
    #              text_annotate_kwargs=dict(
    #                  coord_system='axis',
    #                  text_args=dict(color='white', ha='left', va='top'),
    #                  inset_box_args=None),
    #              **kwargs):
    #     """
    #     make a max-in-pixel density projection using yt via 
    #     yt.load_particles.  note that this method only works with 
    #     yt-3 -- working on a new method for the de-meshed yt-4, but 
    #     it's not quite ready yet

    #     :param positions:
    #         N x 3 array of particle positions to visualize
    #     :param masses:
    #         len N array of particle masses to weight the image
    #     :param bsize:   float
    #         total width (& height & depth) of the box to visualize
    #     :param nref:    int
    #         minimum number of particles to refine on.  takes longer
    #         but yields a higher resolution image with smaller nref
    #     :param img_rotate:  int
    #         either 0, 90, 180, or 270 to rotate the image (in the plane)
    #         through that many degrees.
    #     :param axis:    int or string
    #         axis to project down.  either 0, 1, 2, or 'x', 'y', or 'z'
    #     :pram ptype:    string
    #         type of particle being visualized.  really only needed for the 
    #         labeling.
    #     :param center:  list-like
    #         len 3 array giving the center of the box to visualize
    #     :param cmap: string
    #         valid yt colormap
    #     :param nocbar: bool
    #         set to true to hide the colorbar
    #     :param noaxes:
    #         set to true to hide the axes
    #     :param halos_to_add:
    #         a valid halo catalog with positions and radii of halos 
    #         to indicate as circles
    #     :param fontsize:
    #         size of the font used on the axes + scalebar + etc.
    #     :param scalebar:
    #         size of the scalebar to add, if axes are turned off.  
    #         tries to find a good number on its own if you leave it at None
    #     :param vmin:
    #         minimum scaling of the colorbar.  can also be set to 'min' or None
    #     :param vmax:
    #         maximum scaling of the colorbar.  can also be set to 'max' or None
    #     :param dynamic_range:
    #         dynamic range of the colorbar.  combine with a vmin or vmax, or let 
    #         yt set those then use this to vary vmin using the vmax set by yt
    #     :param circle_radius:
    #         radius of a circle to plot at the center of the image,
    #         (e.g. to indicate Rvir) or None to skip it.
    #     :param circle_args:
    #         a dictionary giving the circle_args param to annotate_circle

    #     kwargs are passed to yt.ProjectionPlot

    #     n.b. if both nocbar and noaxes are true, then release versions of
    #     yt-3.5 will give artifacts in regions where there are no particles.  
    #     to correct this, comment out what is currently line 322 in 
    #     yt/visualization/base_plot_types.py:

    #     in _toggle_axes:
    #     self.axes.set_frame_on(choice) -> #self.axes.set_frame_on(choice)
    #     """
    #     if vmin is not None and vmax is not None and dynamic_range is not None:
    #         raise ValueError(
    #             "can only specify two out of three of vmin, vmax, and dynamic_range")

    #     bl = [-bsize/2, bsize/2]
    #     bbox = np.array([bl, bl, bl])

    #     positions = np.array(positions, copy=True) - center
    #     if points_to_plot is not None:
    #         points_to_plot = np.array(points_to_plot, copy=True) - center

    #     orig_center = np.array(center, copy=True)
    #     center = np.array([0, 0, 0], dtype=float)

    #     msk = boxmsk(positions, center, bl[1])
    #     positions = positions[msk]
    #     masses = masses[msk]

    #     indices, inversions = self.parse_img_rotate(img_rotate, axis)

    #     data = {}
    #     data[(ptype, 'particle_mass')] = masses
    #     data[(ptype, 'particle_position_x')
    #          ] = positions[:, indices[0]]*inversions[0]
    #     data[(ptype, 'particle_position_y')
    #          ] = positions[:, indices[1]]*inversions[1]
    #     data[(ptype, 'particle_position_z')
    #          ] = positions[:, indices[2]]*inversions[2]

    #     if points_to_plot is not None:
    #         points_msk = boxmsk(points_to_plot, center, bl[1])
    #         points_to_plot = points_to_plot[points_msk]

    #         points_to_plot_x = points_to_plot[:, indices[0]]*inversions[0]
    #         points_to_plot_y = points_to_plot[:, indices[1]]*inversions[1]
    #         points_to_plot_z = points_to_plot[:, indices[2]]*inversions[2]

    #     ds = yt.load_particles(data, length_unit=kiloparsec, mass_unit=Msun,
    #                            bbox=bbox, n_ref=nref, periodicity=(False, False, False))

    #     method = kwargs.pop('method', 'mip')

    #     field_name = ptype+'_density'
    #     prj = yt.ProjectionPlot(ds, axis, ('deposit', field_name),
    #                             method=method, center=center, width=(bsize, 'kpc'), **kwargs)

    #     prj = prj.set_figure_size(figure_size)
    #     prj = prj.set_cmap(field_name, cmap)
    #     prj = prj.set_background_color(field_name)

    #     if method == 'mip':
    #         prj = prj.set_unit(field_name, 'Msun/pc**3')
    #     else:
    #         prj = prj.set_unit(field_name, 'Msun/pc**2')

    #     if (vmin is not None) and (vmax is not None):
    #         prj = prj.set_zlim(field_name, vmin, vmax)
    #     elif vmin is not None and dynamic_range is not None:
    #         prj = prj.set_zlim(field_name, zmin=vmin,
    #                            zmax=None, dynamic_range=dynamic_range)
    #     elif vmax is not None and dynamic_range is not None:
    #         prj = prj.set_zlim(field_name, zmin=None,
    #                            zmax=vmax, dynamic_range=dynamic_range)
    #     elif dynamic_range is not None:
    #         zmax = prj.plots[('deposit', field_name)].image.get_clim()[1]
    #         zmin = zmax / dynamic_range
    #         prj = prj.set_zlim(field_name, zmin, zmax)

    #     plot = prj.plots[list(prj.plots)[0]]
    #     ax = plot.axes

    #     if nocbar and noaxes:
    #         buff_size = figure_size * dpi
    #     else:
    #         bounding_box = ax.axes.get_position()
    #         image_size = figure_size * \
    #             max([bounding_box.width, bounding_box.height])
    #         buff_size = image_size * dpi

    #     prj = prj.set_buff_size(buff_size)

    #     img = ax.images[0]
    #     img.set_interpolation(interpolation)

    #     if nocbar:
    #         prj = prj.hide_colorbar()

    #     if noaxes:
    #         prj = prj.hide_axes(draw_frame=True)

    #         # add a scalebar, but note that corner argument doesn't work properly for bigger scale bars, so place by hand
    #         text_args = {'size': fontsize}
    #         if scalebar is None:
    #             minfrac = 0.25
    #             maxfrac = 0.35
    #             pos = (minfrac/2.0 + 0.075, 0.05)
    #             prj = prj.annotate_scale(
    #                 min_frac=minfrac, max_frac=maxfrac, pos=pos, text_args=text_args)
    #         elif isinstance(scalebar, float) or isinstance(scalebar, int):
    #             if scalebar > 0:
    #                 yloc = 0.05
    #                 xloc = ((scalebar/bsize)/2.0 + 0.075)
    #                 pos = (xloc, yloc)
    #                 prj = prj.annotate_scale(
    #                     pos=pos, coeff=scalebar, unit='kpc', text_args=text_args)
    #             else:
    #                 print(
    #                     "Not adding a scalebar to an image with a total width of {} kpc".format(bsize))
    #         else:
    #             print(
    #                 "Not adding a scalebar to an image with a total width of {} kpc".format(bsize))

    #     if halos_to_add is not None:
    #         prj = self.add_halos(prj, halos_to_add, img_rotate=img_rotate,
    #                              axis=axis, center=orig_center, **haloargs)

    #     if circle_radius is not None:
    #         prj = prj.annotate_sphere(center=[0, 0, 0], radius=(
    #             circle_radius, 'kpc'), circle_args=circle_args)

    #     if points_to_plot is not None:
    #         # annotate e.g., star particles on top of the dark matter
    #         print("adding markers for {} points".format(points_to_plot_x.size))
    #         prj = prj.annotate_marker([points_to_plot_x, points_to_plot_y, points_to_plot_z],
    #                                   marker=',', coord_system='data', plot_args={'color': points_color, 's': (72./dpi)**2})  # trying to make the size be a single pixel...

    #     if text_to_annotate is not None:
    #         if 'text_args' not in text_annotate_kwargs:
    #             text_annotate_kwargs['text_args'] = {'fontsize': fontsize}
    #         elif 'fontsize' not in text_annotate_kwargs['text_args']:
    #             text_annotate_kwargs['text_args']['fontsize'] = fontsize
    #         prj = prj.annotate_text(
    #             text_loc, text_to_annotate, **text_annotate_kwargs)

    #     if outname is not None:
    #         prj.save(name=outname, mpl_kwargs=dict(dpi=dpi))
    #     return prj
