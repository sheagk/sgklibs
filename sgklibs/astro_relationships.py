def abundance_match_behroozi_2012(Mhalo, z, alpha=None):
    """
    do abundance matching from arxiv 1207.6105v1
    alpha can be specified as the faint end slope

        * at z = 0, alpha = -1.412 in the original publication
        * at z = 0, GK+2014a argued that alpha = -1.92, which 
            was motivated by the stellar mass function of the
            GAMA survey (Baldry 2012, I believe), yields a 
            much better fit to the smaller dwarf galaxies in
            the Local Group.

    as of 10/2014, this matches with what's on Peter's 
    website
    """

    if alpha is not None:
        vara = True
        if alpha > 0:
            alpha = -1.0*alpha  # make sure relationship is rising
    else:
        vara = False

    from numpy import log10, exp

    def f(x, alpha, delta, gamma):
        top = log10(1+exp(x))**gamma
        bottom = 1 + exp(10**-x)
        return -log10(10**(alpha*x)+1) + delta*top/bottom

    a = 1./(1.+z)

    nu = exp(-4*a**2)
    log10epsilon = -1.777 + (-0.006*(a-1) - 0.000*z)*nu - 0.119*(a-1)
    epsilon = 10**log10epsilon

    log10M1 = 11.514 + (-1.793*(a-1) - 0.251*z)*nu
    M1 = 10**log10M1

    if alpha is None:
        alpha = -1.412 + (0.731*(a-1))*nu
    else:
        defalpha = -1.412 + (0.731*(a-1))*nu

    delta = 3.508 + (2.608*(a-1) - 0.043*z)*nu
    gamma = 0.316 + (1.319*(a-1) + 0.279*z)*nu

    if not vara:
        log10Mstar = log10(epsilon*M1) + f(log10(Mhalo/M1),
                                           alpha, delta, gamma) - f(0, alpha, delta, gamma)

    else:
        from numpy import array, empty_like
        if type(Mhalo) != type(array([1.0, 2.0, 3.0])):
            if Mhalo >= M1:
                # then I use the default alpha
                log10Mstar = log10(epsilon*M1) + f(log10(Mhalo/M1),
                                                   defalpha, delta, gamma) - f(0, defalpha, delta, gamma)
            else:
                # then I use my alpha
                log10Mstar = log10(epsilon*M1) + f(log10(Mhalo/M1),
                                                   alpha, delta, gamma) - f(0, alpha, delta, gamma)
        else:
            log10Mstar = empty_like(Mhalo)
            log10Mstar[Mhalo >= M1] = log10(epsilon*M1) + f(log10(
                Mhalo[Mhalo >= M1]/M1), defalpha, delta, gamma) - f(0, defalpha, delta, gamma)
            log10Mstar[Mhalo < M1] = log10(
                epsilon*M1) + f(log10(Mhalo[Mhalo < M1]/M1), alpha, delta, gamma) - f(0, alpha, delta, gamma)

    return 10**log10Mstar


def bryan_norman_virial_ciritcal_overdensity(redshift, Om0, Ode0):
    """
    returns the virial overdensity at a given redshift for a given 
    cosmology from Bryan & Norman 1998 (I think)
    """

    import numpy as np
    z_plus_one = 1 + redshift
    Esq = Om0 * z_plus_one**3 + Ode0
    Omega = Om0 * z_plus_one**3 / Esq
    x = Omega - 1
    return 18*(np.pi**2) + 82*x - 39*(x**2)
