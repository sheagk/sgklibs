from .low_level_utils import backup_file, isint, isfloat, boxmsk
from .particle_io import read_part_wetzel
import utilities as ut
import numpy as np
import matplotlib
matplotlib.use('agg', warn=False)


# import Andrew Wetzel's utilities package


class ProjectParticlesYT():
    def __init__(self, positions, masses,
                 outname=None,
                 bsize=800, nref=8,
                 img_rotate=0, axis=0,
                 ptype='dark', center=np.array([0, 0, 0]),
                 cmap='cubehelix',
                 nocbar=True, noaxes=True,
                 halos_to_add=None, haloargs={},
                 fontsize=28, interpolation='bicubic',
                 scalebar=None,
                 vmin=None, vmax=None, dynamic_range=None,
                 circle_radius=None, circle_args={},
                 points_to_plot=None, points_color='red',
                 number_density=False,
                 dpi=300, figure_size=10,
                 text_to_annotate=None,
                 text_loc=(0.025, 0.975),
                 text_annotate_kwargs=dict(
                     coord_system='axis',
                     text_args=dict(color='white', ha='left', va='top'),
                     inset_box_args=None),
                 **kwargs):
        """
        make a max-in-pixel density projection using yt via 
        yt.load_particles.  note that this method only works with 
        yt-3 -- working on a new method for the de-meshed yt-4, but 
        it's not quite ready yet

        :param positions:
            N x 3 array of particle positions to visualize
        :param masses:
            len N array of particle masses to weight the image
        :param bsize:   float
            total width (& height & depth) of the box to visualize
        :param nref:    int
            minimum number of particles to refine on.  takes longer
            but yields a higher resolution image with smaller nref
        :param img_rotate:  int
            either 0, 90, 180, or 270 to rotate the image (in the plane)
            through that many degrees.
        :param axis:    int or string
            axis to project down.  either 0, 1, 2, or 'x', 'y', or 'z'
        :pram ptype:    string
            type of particle being visualized.  really only needed for the 
            labeling.
        :param center:  list-like
            len 3 array giving the center of the box to visualize
        :param cmap: string
            valid yt colormap
        :param nocbar: bool
            set to true to hide the colorbar
        :param noaxes:
            set to true to hide the axes
        :param halos_to_add:
            a valid halo catalog with positions and radii of halos 
            to indicate as circles
        :param fontsize:
            size of the font used on the axes + scalebar + etc.
        :param scalebar:
            size of the scalebar to add, if axes are turned off.  
            tries to find a good number on its own if you leave it at None
        :param vmin:
            minimum scaling of the colorbar.  can also be set to 'min' or None
        :param vmax:
            maximum scaling of the colorbar.  can also be set to 'max' or None
        :param dynamic_range:
            dynamic range of the colorbar.  combine with a vmin or vmax, or let 
            yt set those then use this to vary vmin using the vmax set by yt
        :param circle_radius:
            radius of a circle to plot at the center of the image,
            (e.g. to indicate Rvir) or None to skip it.
        :param circle_args:
            a dictionary giving the circle_args param to annotate_circle

        kwargs are passed to yt.ProjectionPlot

        n.b. if both nocbar and noaxes are true, then release versions of
        yt-3.5 will give artifacts in regions where there are no particles.  
        to correct this, comment out what is currently line 322 in 
        yt/visualization/base_plot_types.py:

        in _toggle_axes:
        self.axes.set_frame_on(choice) -> #self.axes.set_frame_on(choice)
        """
        if vmin is not None and vmax is not None and dynamic_range is not None:
            raise ValueError(
                "can only specify two out of three of vmin, vmax, and dynamic_range")

        bl = [-bsize/2, bsize/2]
        bbox = np.array([bl, bl, bl])

        positions = np.array(positions, copy=True) - center
        if points_to_plot is not None:
            points_to_plot = np.array(points_to_plot, copy=True) - center

        orig_center = np.array(center, copy=True)
        center = np.array([0, 0, 0], dtype=float)

        msk = boxmsk(positions, center, bl[1])
        positions = positions[msk]
        masses = masses[msk]

        indices, inversions = self.parse_img_rotate(img_rotate, axis)

        data = {}
        data[(ptype, 'particle_mass')] = masses
        data[(ptype, 'particle_position_x')
             ] = positions[:, indices[0]]*inversions[0]
        data[(ptype, 'particle_position_y')
             ] = positions[:, indices[1]]*inversions[1]
        data[(ptype, 'particle_position_z')
             ] = positions[:, indices[2]]*inversions[2]

        if points_to_plot is not None:
            points_msk = boxmsk(points_to_plot, center, bl[1])
            points_to_plot = points_to_plot[points_msk]

            points_to_plot_x = points_to_plot[:, indices[0]]*inversions[0]
            points_to_plot_y = points_to_plot[:, indices[1]]*inversions[1]
            points_to_plot_z = points_to_plot[:, indices[2]]*inversions[2]

        ds = yt.load_particles(data, length_unit=kiloparsec, mass_unit=Msun,
                               bbox=bbox, n_ref=nref, periodicity=(False, False, False))

        method = kwargs.pop('method', 'mip')

        field_name = ptype+'_density'
        prj = yt.ProjectionPlot(ds, axis, ('deposit', field_name),
                                method=method, center=center, width=(bsize, 'kpc'), **kwargs)

        prj = prj.set_figure_size(figure_size)
        prj = prj.set_cmap(field_name, cmap)
        prj = prj.set_background_color(field_name)

        if method == 'mip':
            prj = prj.set_unit(field_name, 'Msun/pc**3')
        else:
            prj = prj.set_unit(field_name, 'Msun/pc**2')

        if (vmin is not None) and (vmax is not None):
            prj = prj.set_zlim(field_name, vmin, vmax)
        elif vmin is not None and dynamic_range is not None:
            prj = prj.set_zlim(field_name, zmin=vmin,
                               zmax=None, dynamic_range=dynamic_range)
        elif vmax is not None and dynamic_range is not None:
            prj = prj.set_zlim(field_name, zmin=None,
                               zmax=vmax, dynamic_range=dynamic_range)
        elif dynamic_range is not None:
            zmax = prj.plots[('deposit', field_name)].image.get_clim()[1]
            zmin = zmax / dynamic_range
            prj = prj.set_zlim(field_name, zmin, zmax)

        plot = prj.plots[list(prj.plots)[0]]
        ax = plot.axes

        if nocbar and noaxes:
            buff_size = figure_size * dpi
        else:
            bounding_box = ax.axes.get_position()
            image_size = figure_size * \
                max([bounding_box.width, bounding_box.height])
            buff_size = image_size * dpi

        prj = prj.set_buff_size(buff_size)

        img = ax.images[0]
        img.set_interpolation(interpolation)

        if nocbar:
            prj = prj.hide_colorbar()

        if noaxes:
            prj = prj.hide_axes(draw_frame=True)

            # add a scalebar, but note that corner argument doesn't work properly for bigger scale bars, so place by hand
            text_args = {'size': fontsize}
            if scalebar is None:
                minfrac = 0.25
                maxfrac = 0.35
                pos = (minfrac/2.0 + 0.075, 0.05)
                prj = prj.annotate_scale(
                    min_frac=minfrac, max_frac=maxfrac, pos=pos, text_args=text_args)
            elif isinstance(scalebar, float) or isinstance(scalebar, int):
                if scalebar > 0:
                    yloc = 0.05
                    xloc = ((scalebar/bsize)/2.0 + 0.075)
                    pos = (xloc, yloc)
                    prj = prj.annotate_scale(
                        pos=pos, coeff=scalebar, unit='kpc', text_args=text_args)
                else:
                    print(
                        "Not adding a scalebar to an image with a total width of {} kpc".format(bsize))
            else:
                print(
                    "Not adding a scalebar to an image with a total width of {} kpc".format(bsize))

        if halos_to_add is not None:
            prj = self.add_halos(prj, halos_to_add, img_rotate=img_rotate,
                                 axis=axis, center=orig_center, **haloargs)

        if circle_radius is not None:
            prj = prj.annotate_sphere(center=[0, 0, 0], radius=(
                circle_radius, 'kpc'), circle_args=circle_args)

        if points_to_plot is not None:
            # annotate e.g., star particles on top of the dark matter
            print("adding markers for {} points".format(points_to_plot_x.size))
            prj = prj.annotate_marker([points_to_plot_x, points_to_plot_y, points_to_plot_z],
                                      marker=',', coord_system='data', plot_args={'color': points_color, 's': (72./dpi)**2})  # trying to make the size be a single pixel...

        if text_to_annotate is not None:
            if 'text_args' not in text_annotate_kwargs:
                text_annotate_kwargs['text_args'] = {'fontsize': fontsize}
            elif 'fontsize' not in text_annotate_kwargs['text_args']:
                text_annotate_kwargs['text_args']['fontsize'] = fontsize
            prj = prj.annotate_text(
                text_loc, text_to_annotate, **text_annotate_kwargs)

        if outname is not None:
            prj.save(name=outname, mpl_kwargs=dict(dpi=dpi))
        return prj

    def add_halos(self, prj, hal,
                  center=np.array([0, 0, 0]),
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

        hal_positions = hal.prop(position_prop) - center
        indices, inversions = self.parse_img_rotate(img_rotate, axis)

        import rockstar_analysis

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
            prj.annotate_sphere(c, radius=(
                hal[radiusprop][idx], 'kpc'), circle_args=circle_args)
        return prj

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
