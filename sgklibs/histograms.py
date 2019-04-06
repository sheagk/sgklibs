#!/usr/bin/env python3


def cumulative_histogram(data, weights=None, bins=None, normed=False, zero=True):
    """
    Makes an cumulative histogram (e.g. N(<=M), M(<t), etc.) when fed data 
    (e.g. M values in the first case or t values w/ M weights in the second).

    :param data:
        the input array to be made into a histogram, already presliced etc.
        this is basically the x-values of your items

    :param weights:
        weights to assign to the values.  defaults to 1 for everything

    :bool normed:
        length of the dataset
        if true, normalized to max out at 1.  Otherwise, goes from 1 to the
        length of the dataset

    :param bins:
        if given, used as the bins or number of bins for the histogram
        if not given, the unique
        values in the data are used as the histogram (so the returned histogram
        is exact in that case)

    :param zero:
        adds a zero to the beginning of the returned y array (w/ the same xvalue)

    :returns:
        the histogram, then the bins, so to plot, do, e.g.
        hist,bins = cumulative_histogram(vmax[slice])
        loglog(bins,hist)
    """
    import numpy as np

    if len(data) == 0:
        print("Passed no data; returning None for the hist and bins")
        return None, None
    if type(data) != type(np.array([0, 1, 2])):
        data = np.array(data)
    if weights is None:
        weights = np.ones(data.size, dtype=float)
    elif np.isscalar(weights):
        weights = np.array([weights]*data.size, dtype=float)
    else:
        assert weights.shape == data.shape

    if bins is not None:  # then this problem is pretty easy
        if np.isscalar(bins):
            xmin = data.min()
            xmax = data.max()

            if xmax / xmin >= 300:
                bins = np.logspace(np.log10(xmin), np.log10(xmax), int(bins))
            else:
                bins = np.linspace(xmin, xmax, int(bins))
        else:
            bins = np.sort(np.array(bins))

        ## hist = empty(bins.size,dtype=float)
        # for ii in range(bins.shape[0]):
        ##     hist[ii] = weights[data<=bins[ii]].sum()
        # hist = np.array([weights[data<=b].sum() for b in bins])
        counts, bin_edges = np.histogram(data, bins=bins, weights=weights)
        hist = np.cumsum(counts)
        bins = bin_edges[1:]

    else:
        sorti = np.argsort(data)

        data = data[sorti]
        weights = weights[sorti]

        # now, I don't want to do just cumsum because I'm worried about duplicate xvalues
        # so let's sum up the weights that fall into any individual bins
        bins, unique_indices, appearances = np.unique(
            data, return_index=True, return_counts=True)

        output_weights = np.empty(bins.size, dtype=weights.dtype)
        unq = appearances == 1
        not_unq = np.logical_not(unq)

        # values that appear once are easy -- just have to copy over their weight
        output_weights[unq] = weights[unique_indices[unq]]

        # values that appear multiple times are harder -- sum up the weights for each value
        lefts = np.searchsorted(data, bins[not_unq], side='left')
        rights = np.searchsorted(data, bins[not_unq], side='right')
        not_unq_indices = np.arange(bins.size, dtype=int)[not_unq]
        for ii in range(not_unq_indices.size):
            l = lefts[ii]
            r = rights[ii]
            idx = not_unq_indices[ii]
            if r == weights.size:  # then I just go to the end of the array
                output_weights[idx] = np.sum(weights[l:])
            else:
                output_weights[idx] = np.sum(weights[l:r])

        # now I can sum up the weights
        hist = np.cumsum(output_weights)

    if normed:
        hist = hist*1.0/hist[-1]  # max it out at 1

    if zero:
        # want the first entry to be 0 so that the lines don't start in the middle of the plot
        # make it just a little bigger than zero to work with log plots
        hist = np.concatenate(([1e-10], hist))
        bins = np.concatenate(([bins[0]-1e-10], bins))

    # returns y, x
    return hist, bins


def anticum_hist(data, bins=None, toadd=0, zero=False, zerox=False):
    """
    Makes an anticumulative histogram (e.g. N(>Vmax)) when fed data (e.g. Vmax
    values).

    :param data:
        the input array to be made into a histogram, already presliced etc.

    :param bins:
        if given, used as the bins for the histogram; if not given, the unique
        values in the data are used as the histogram (so the returned histogram
        is exact in that case)

    :param zero: bool
        whether or not to append 1e-10 to the counts, so that the line appears
        to go to zero.  requires you to set y-limits by hand when plotting.

    :param zerox: bool
        whether or not to prepent 1e-10 to x data so that the line appears to go
        to zero there too.

    :param toadd:
        add this number to all the y values, e.g. if there are known objects
        not being included in the histogram (for example, for sigma counts 
        when you have the LMC)

    :returns:
        the histogram, then the bins, so to plot, do, e.g.
        hist,bins = anticum_hist(vmax[slice])
        loglog(bins,hist)
    """

    from numpy import histogram, unique, append, cumsum, zeros, array
    data = array(data)

    if bins is None:
        temp, bins = histogram(data, unique(data))
        hist = append(cumsum(temp[::-1])[::-1], 1)
    else:
        if len(data) == 0:
            return zeros(len(bins)), bins
        # if max(data) > max(bins):
        #     print "Must have the largest bin be bigger than the largest data point."
        #     print max(bins)
        #     print max(data)
        #     import sys
        #     sys.exit(1337)
        temp, bins = histogram(data, bins=bins)
        numbig = data[data > bins.max()].shape[0]
        # add on the objects that are above the last bin to the last count, so that they're included in the cumulative sum
        temp[-1] += numbig
        hist = append(cumsum(temp[::-1])[::-1], numbig)

    hist = hist + toadd  # add some number to all the values

    if zero:
        hist = append(hist, 1e-10)
        bins = append(bins, bins[-1])
    if zerox:
        hist = append(hist[0], hist)
        bins = append(1e-10, bins)

    return hist, bins
