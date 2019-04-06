#!/usr/bin/env

import numpy as np
from scipy import stats as st
from tqdm import tqdm

from .low_level_utils import normalize_array


def permutation_pearson(xvals, yvals, trials=1e5, normalize=True, doparallel=True):
    """
    given a list of xvals and yvals, returns a p-value that is
    the fraction of the trials that return a larger (in magnitude) 
    Pearson r coefficient than the actual x-y pairs.  smaller
    p-value means more correlation exists (that is lost when you
    scramble).
    """
    try:
        from joblib import Parallel, delayed
        try:
            from psutil import cpu_count
        except ImportError:
            from multiprocessing import cpu_count
        njobs = cpu_count()
    except ImportError:
        doparallel = False

    trials = int(trials)

    from scipy.stats import pearsonr

    xvals = np.array(xvals)
    yvals = np.array(yvals)
    if normalize:
        xvals = normalize_array(xvals)
        yvals = normalize_array(yvals)

    actual_r, actual_p = st.pearsonr(xvals, yvals)
    npts = yvals.size
    if doparallel:
        rvalues = Parallel(n_jobs=njobs)(delayed(pearsonr)(
            xvals, np.random.permutation(yvals)) for ii in range(trials))
        rvalues = np.array(rvalues)[:, 0]
    else:
        rvalues = np.array([pearsonr(xvals, np.random.permutation(yvals))[0] for ii in range(
            trials)])  # compute the r value for n trials of x vs randomly sorting y
    # count the number of trials that have a larger (in magnitude) r value than the actual sorting
    nbetter = np.count_nonzero(np.abs(rvalues) > np.abs(actual_r))
    return nbetter*1.0/trials


def permutation_spearman(xvals, yvals, trials=1e5, normalize=True, doparallel=True):
    """
    given a list of xvals and yvals, returns a p-value that is
    the fraction of the trials that return a larger (in magnitude) 
    Spearmann r coefficient than the actual x-y pairs.  smaller
    p-value means more correlation exists (that is lost when you
    scramble).
    """
    try:
        from joblib import Parallel, delayed
        try:
            from psutil import cpu_count
        except ImportError:
            from multiprocessing import cpu_count
        njobs = cpu_count()
    except ImportError:
        doparallel = False

    from scipy.stats import spearmanr
    trials = int(trials)

    xvals = np.array(xvals)
    yvals = np.array(yvals)

    if normalize:
        xvals = normalize_array(xvals)
        yvals = normalize_array(yvals)

    actual_r, actual_p = st.spearmanr(xvals, yvals)
    npts = yvals.size
    if doparallel:
        rvalues = Parallel(n_jobs=njobs)(delayed(spearmanr)(
            xvals, np.random.permutation(yvals)) for ii in range(trials))
        rvalues = np.array(rvalues)[:, 0]
    else:
        rvalues = np.array([spearmanr(xvals, np.random.permutation(yvals))[0] for ii in range(
            trials)])  # compute the r value for n trials of x vs randomly sorting y
    # count the number of trials that have a larger (in magnitude) rank coeffecient than the actual sorting
    nbetter = np.count_nonzero(np.abs(rvalues) > np.abs(actual_r))
    return nbetter*1.0/trials


def bootstrap_pearson(xvals, yvals, trials=1e5, normalize=True, CI=95.):
    """
    given a list of xvals and yvals, returns a 95% confidence interval
    that is calculated from the distribution of p-values that you get 
    if you draw len(``xvals'') pairs from the actual data trials times.

    just like actual pearson p-value, smaller is better
    """
    trials = int(trials)
    xvals = np.array(xvals)
    yvals = np.array(yvals)

    if normalize:
        xvals = normalize_array(xvals)
        yvals = normalize_array(yvals)

    npts = yvals.size
    rvalues = np.zeros(trials, dtype='f')

    for ii in tqdm(range(trials)):
        idx = np.random.choice(npts, size=npts, replace=True)
        r, p = st.pearsonr(xvals[idx], yvals[idx])
        rvalues[ii] = r

    return np.percentile(rvalues, [50-CI/2., 50+CI/2.])


def bootstrap_spearman(xvals, yvals, trials=1e5, CI=95., returndata=False):
    """
    given a list of xvals and yvals, returns a 95% confidence interval
    that is calculated from the distribution of p-values that you get 
    if you draw len(``xvals'') pairs from the actual data trials times.

    just like actual spearman p-value, smaller is better correlation
    """
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    trials = int(trials)

    npts = yvals.size
    rvalues = np.zeros(trials, dtype=float)

    for ii in tqdm(range(int(trials))):
        idx = np.random.choice(npts, size=npts, replace=True)
        r, p = st.spearmanr(xvals[idx], yvals[idx])
        rvalues[ii] = r

    ret = np.percentile(rvalues, [50-CI/2., 50+CI/2.])
    if returndata:
        return ret[0], ret[1], rvalues
    return ret


def bootstrap_ADksamp(samples, CI=95., trials=1e5, warn=False):
    """
    given a list of samples, where each entry in that list is a collection of
    observations/values, returns the lower and upper ends confidence interval 
    (calculated by boostrapping the samples over some number of trials) of the 
    k-sample Anderson-Darling statistic (scipy.stats.anderson_ksamp), which 
    measures how likely it is that the k samples were drawn from the same 
    underlying population.  

    returns two lists:
        first, you get the lower and upper bounds of the CI on the AD statistics 
            (straight from from scipy.stats.anderson_ksamp).
        second, you get the p values, which depend on the statistic generated and 
            the number of samples being compared (not quite straight from 
            scipy.stats.anderson_ksamp, but using code copied from there)
    """

    samples = list(map(np.asarray, samples))
    lengths = [v.size for v in samples]
    trials = int(trials)

    output_stats = np.empty(trials, dtype=float)
    for ii in tqdm(range(trials)):
        this_samples = [
            v[np.random.choice(n, size=n, replace=True)] for v, n in zip(samples, lengths)]
        stat, critical, siglevel = st.anderson_ksamp(this_samples)
        output_stats[ii] = stat

    # now get the output stats that correspond to the requested CI:
    lower, upper = np.percentile(output_stats, [50-CI/2., 50+CI/2.])

    # now interpolate/extrapolate the significance level of the lower/upper:
    # crit_values are the same for every iteration because it just depends on
    #   len(this_samples), which == len(samples), which never changes
    # following is copied straight out of scipy's source for anderson_ksamp
    pf = np.polyfit(critical, np.log(
        np.array([0.25, 0.1, 0.05, 0.025, 0.01])), 2)
    if warn:
        if lower < critical.min() or lower > critical.max():
            print("!! -- warning:  lower significance lever will be extrapolated")
        if upper < critical.min() or upper > critical.max():
            print("!! -- warning:  upper significance lever will be extrapolated")

    p_lower = np.exp(np.polyval(pf, lower))
    p_upper = np.exp(np.polyval(pf, upper))
    return [lower, upper], [p_lower, p_upper]
