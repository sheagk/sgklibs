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
