#!/usr/bin/env python3

import numpy as np


def median_in_rolling_bins(x_ar, y_ar, pts_per_bin=15):
    '''
    compute the median of a set of points (given by 
    x_ar[ii], y_ar[ii]) using a fixed number of points
    per bin, starting at the smallest x value.  returns 
    the median x and y values of the points in each bin.
    '''

    # sort the values by their x prop
    sorti = np.argsort(x_ar)

    # make sure that they're arrays; otherwise, can't sort with inidces
    sorted_x = np.asarray(x_ar)[sorti]
    sorted_y = np.asarray(y_ar)[sorti]

    nbins = int(np.rint(np.ceil(sorted_x.size/pts_per_bin)))
    median_x = np.empty(nbins)
    median_y = np.empty(nbins)

    left = 0
    for ii in range(nbins):
        right = min([left + pts_per_bin, sorted_x.size])

        bx = sorted_x[left:right]
        by = sorted_y[left:right]

        median_x[ii] = np.median(bx)
        median_y[ii] = np.median(by)

        left = right

    return median_x, median_y


def average_lines_linear(xvals, yvals, output_xvals, scatter,
                         do_median, percentile_interp='linear'):
    '''
    take the average (in linear space) of a group of lines at output_xvals 
    and measure the scatter (if > 0).  does the median if do_median.

    does the averaging by piecewise linearly interpolating (with np.interp)
    the lines passed in.  lines are defined as xvals[ii] vs yvals[ii]

    extends the lines outside their xrange by assuming they go flat on 
    either end (which makes sense for averaging cumulative histograms)
    '''

    import numpy as np
    assert type(output_xvals) == np.ndarray

    lines_at_output_x = []
    for ii in range(len(xvals)):
        xv = np.array(xvals[ii])

        # make sure they're sorted in increasing x
        sorti = np.argsort(xv)
        yv = np.array(yvals[ii])[sorti]
        xv = xv[sorti]

        # extend out each line to go from the the smallest xval to the largest xval w/ a flat value
        xv = np.concatenate(([min([output_xvals[0], xv.min()-1e-10])],
                             xv, [max([output_xvals[-1], xv.max()+1e-10])]))
        yv = np.concatenate(([yv[0]], yv, [yv[-1]]))

        lines_at_output_x.append(np.interp(output_xvals, xv, yv))

    # lines_at_output_x = [np.interp(output_xvals, xvals[ii], yvals[ii]) for ii in range(len(xvals))]
    average_function = np.nanmedian if do_median else np.nanmean

    output_median = np.ones(output_xvals.size)*-1
    output_low = np.ones(output_xvals.size)*-1
    output_high = np.ones(output_xvals.size)*-1

    for jj, xv in enumerate(output_xvals):
        yvs = np.array([lines_at_output_x[ii][jj]
                        for ii in range(len(lines_at_output_x))])
        if scatter == False or scatter <= 0:
            # not doing scatter, so just take the average with whichver function I'm supposed do
            l = m = h = average_function(yvs)
        elif do_median:
            # am doing scatter, and also doing a median
            l, m, h = np.nanpercentile(
                yvs, [50 - scatter/2, 50, 50+scatter/2], interpolation=percentile_interp)
        else:
            # am doing scatter, not doing median, so doing a mean + scatter
            l, h = np.nanpercentile(
                yvs, [50 - scatter/2, 50+scatter/2], interpolation=percentile_interp)
            m = np.nanmean(yvs)
        output_low[jj] = l
        output_median[jj] = m
        output_high[jj] = h
    return {'x': output_xvals, 'median': output_median, 'low': output_low, 'high': output_high}


def average_lines_cubic(xvals, yvals, output_xvals, scatter, extrapolate, do_median,
                        percentile_interp='linear'):
    '''
    take the average (in linear space) of a group of lines at output_xvals 
    and measure the scatter (if > 0).  does the median if do_median.

    does the averaging by piecewise linearly interpolating (with np.interp)
    the lines passed in.  lines are defined as xvals[ii] vs yvals[ii]

    extends the lines outside their xrange by assuming they go flat on 
    either end (which makes sense for averaging cumulative histograms)
    '''

    assert extrapolate in [False, True, 'flat']
    from scipy.interpolate import interp1d
    import numpy as np

    assert type(output_xvals) == np.ndarray

    if extrapolate in ['flat', False]:
        fill_value = np.nan
    else:
        fill_value = 'extrapolate'

    functions = []
    for ii in range(len(xvals)):
        xv, yv = xvals[ii], yvals[ii]
        if extrapolate == 'flat':
            # append the min and max ys at the min and max output_xvals
            xv = np.concatenate(([min([output_xvals[0], xv.min()-1e-10])],
                                 xv, [max([output_xvals[-1], xv.max()+1e-10])]))
            yv = np.concatenate(([yv[0]], yv, [yv[-1]]))
        functions.append(interp1d(xv, yv, kind='cubic', bounds_error=False,
                                  fill_value=fill_value, assume_sorted=True))
    # functions = [interp1d(xvals[ii], yvals[ii], kind='cubic', bounds_error=False, fill_value=fill_value, assume_sorted=True) for ii in range(len(xvals))]     #don't raise errors; just return nan's

    average_function = np.nanmedian if do_median else np.nanmean

    if scatter <= 0 or scatter == False:
        # then ignore those nan's when taking the mean
        output_yvals = np.array(
            [average_function([f(xv) for f in functions]) for xv in output_xvals])
        return {'x': output_xvals, 'median': output_yvals}

    output_median = np.empty_like(output_xvals)
    output_low = np.empty_like(output_xvals)
    output_high = np.empty_like(output_xvals)

    for ii, xv in enumerate(output_xvals):
        yvs = np.array([f(xv) for f in functions])
        if do_median:
            l, m, h = np.nanpercentile(
                yvs, [50 - scatter/2, 50, 50+scatter/2], interpolation=percentile_interp)
        else:
            l, h = np.nanpercentile(
                yvs, [50 - scatter/2, 50+scatter/2], interpolation=percentile_interp)
            m = np.nanmean(yvs)
        output_low[ii] = l
        output_median[ii] = m
        output_high[ii] = h
    return {'x': output_xvals, 'median': output_median, 'low': output_low, 'high': output_high}


def average_lines(xvals, yvals, output_xvals=None, scatter=0, do_median=True,
                  linear_interp=False, extrapolate=False, percentile_interp='linear'):
    """
    get the median (in y) along with the scatter (if scatter_percentile > 0) 
    of a bunch of lines defined as xvals[ii] vs yvals[ii].  output_xvals can 
    be either an integer or a list-like object of values

    does the median+scatter by doing a cubic spline for each line
    """

    import numpy as np

    assert len(xvals) == len(yvals)
    len1 = False
    for ii in range(len(xvals)):
        if np.isscalar(xvals[ii]):
            len1 = True
        else:
            assert len1 == False
            assert len(xvals[ii]) == len(yvals[ii])

    if len1:
        # if I hadned in a len1 array/list, (i.e., a single line), then there's no scatter (low = high = med)
        return {'x': this_xvals[msk], 'median': this_yvals[msk], 'low': this_yvals[msk], 'high': this_yvals[msk]}

    if output_xvals is None or type(output_xvals) == int:
        xmin = min([min(xv) for xv in xvals])
        xmax = max([max(xv) for xv in xvals])
        if output_xvals is None:
            output_xvals = 1000

        if xmin == 0:
            output_xvals = np.linspace(xmin, xmax, output_xvals)
        elif xmax / xmin >= 300:
            output_xvals = np.logspace(
                np.log10(xmin), np.log10(xmax), output_xvals)
        else:
            output_xvals = np.linspace(xmin, xmax, output_xvals)
    else:
        output_xvals = np.sort(output_xvals)

    if linear_interp:
        return average_lines_linear(xvals, yvals, output_xvals=output_xvals, scatter=scatter, do_median=do_median, percentile_interp=percentile_interp)
    else:
        return average_lines_cubic(xvals, yvals, output_xvals=output_xvals, scatter=scatter, extrapolate=extrapolate, do_median=do_median, percentile_interp=percentile_interp)


def average_lines_noscatter(xvals, yvals, output_xvals=None, extrapolate=False):
    """
    get the average (in y) of a bunch of lines defined
    as xvals[ii] vs yvals[ii].  output_xvals can be
    either an integer or a list-like object of values

    does the averaging by doing a cubic spline for each
    line
    """

    import numpy as np
    from scipy.interpolate import interp1d

    assert len(xvals) == len(yvals)
    for ii in range(len(xvals)):
        assert len(xvals[ii]) == len(yvals[ii]) == len(xvals[0])

    if output_xvals is None or type(output_xvals) == int:
        xmin = min([min(xv) for xv in xvals])
        xmax = max([max(xv) for xv in xvals])
        if output_xvals is None:
            output_xvals = 1000

        if xmin == 0:
            output_xvals = np.linspace(xmin, xmax, output_xvals)
        elif xmax / xmin >= 300:
            output_xvals = np.logspace(
                np.log10(xmin), np.log10(xmax), output_xvals)
        else:
            output_xvals = np.linspace(xmin, xmax, output_xvals)

    functions = [interp1d(xvals[ii], yvals[ii], kind='cubic', bounds_error=False, fill_value='extrapolate' if extrapolate else np.nan)
                 for ii in range(len(xvals))]  # don't raise errors; just return nan's
    # then ignore those nan's when taking the mean
    output_yvals = np.array(
        [np.nanmean([f(xv) for f in functions]) for xv in output_xvals])
    return output_xvals, output_yvals
