# Adapted from https://openreview.net/attachment?id=l0V53bErniB&name=supplementary_material


import math

invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2


def gss(f, args, a, b, dev, tol=1e-5):
    '''Golden section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    modified from: https://en.wikipedia.org/wiki/Golden-section_search

    Usage: gss(f_gss, [G_reg_2, ups, super_regions, updates_proj, .5, False, recall], 0., 1., torch.device)
    '''

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c, args, dev)
    yd = f(d, args, dev)

    for k in range( n -1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c, args, dev)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d, args, dev)

    if yc < yd:
        return (a, d)
    else:
        return (c, b)
