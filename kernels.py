import math
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

quintic_norm_1d = 1/120
quintic_norm_2d = 7/(478*math.pi)
quintic_norm_3d = 1/(120*math.pi)

def w_gauss( r, h, dim ):
    # from phantom sph: https://pysph.readthedocs.io/en/latest/_modules/pysph/base/kernels.html
    norm_const = 1 / math.sqrt(np.pi)

    h1 = 1.0 / h

    q = r * h1
    
    #w = ( (h1 * norm_const) **dim )  * math.exp( -r**2 / h**2)
    w = ( (h1 * norm_const) **dim )  * np.exp( -q*q)
    
    return w

def dwdq_gauss(r, h, dim):
    # from phantom sph: https://pysph.readthedocs.io/en/latest/_modules/pysph/base/kernels.html
    norm_const = 1 / math.sqrt(math.pi)
    h1 = 1.0 / h
    q = r * h1

    return ((h1*norm_const)**dim) * -2.0 * q * np.exp(-q * q)

def gradw_gauss(r, h, dim):
    h1 = 1. / h

    # compute the gradient.
    if (r > 1e-12):
        wdash = dwdq_gauss(r, h, dim)
        tmp = wdash * h1 / r
    else:
        tmp = 0.0

    return tmp

def w_quintic(r, h, dim):
    # from phantom sph: https://pysph.readthedocs.io/en/latest/_modules/pysph/base/kernels.html
    if dim == 1:
        sigma = quintic_norm_1d
    elif dim == 2:
        sigma = quintic_norm_2d
    else:
        sigma = quintic_norm_3d

    h1 = 1. / h
    q = r * h1
    
    tmp3 = 3. - q
    tmp2 = 2. - q
    tmp1 = 1. - q

    val1 = tmp3 * tmp3 * tmp3 * tmp3 * tmp3
    val1 -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2

    val0 = tmp3 * tmp3 * tmp3 * tmp3 * tmp3
    val0 -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2
    val0 += 15. * tmp1 * tmp1 * tmp1 * tmp1 * tmp1

    if q >= 3:
        w = 0.0
    elif q >= 2:
        w = tmp3 * tmp3 * tmp3 * tmp3 * tmp3
    elif q >= 1:
        w = val1
    else:
        w = val0

    return (h1 ** dim) * sigma * w


def dwdq_quintic(r, h, dim):
    # from phantom sph: https://pysph.readthedocs.io/en/latest/_modules/pysph/base/kernels.html
    if dim == 1:
        sigma = quintic_norm_1d
    elif dim == 2:
        sigma = quintic_norm_2d
    else:
        sigma = quintic_norm_3d

    h1 = 1. / h
    q = r * h1
    
    tmp3 = 3. - q
    tmp2 = 2. - q
    tmp1 = 1. - q

    val1 = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
    val1 += 30.0 * tmp2 * tmp2 * tmp2 * tmp2

    val0 = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
    val0 += 30.0 * tmp2 * tmp2 * tmp2 * tmp2
    val0 -= 75.0 * tmp1 * tmp1 * tmp1 * tmp1

    if q >= 3:
        grad_w = 0.0
    elif q >= 2:
        grad_w = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
    elif q >= 1:
        grad_w = val1
    else:
        grad_w = val0

    if r > 1e-12:
        return (h1 ** dim) * sigma * grad_w
    else:
        return 0.0

def gradw_quintic(r, h, dim):
    h1 = 1. / h

    # compute the gradient.
    if (r > 1e-12):
        wdash = dwdq_quintic(r, h, dim)
        tmp = wdash * h1 / r
    else:
        tmp = 0.0

    return tmp

#w_gauss_gpu = cuda.jit("float32(float32, float32, int64)", device=True)(w_gauss)
#gradw_gauss_gpu = cuda.jit("float32(float32, float32, int64)", device=True)(gradw_gauss)
#dwdq_gauss_gpu = cuda.jit("float32(float32, float32, int64)", device=True)(dwdq_gauss)

w_quintic_gpu = cuda.jit("float32(float32, float32, int64)", device=True)(w_quintic)
#gradw_quintic_gpu = cuda.jit("float32(float32, float32, int64)", device=True)(gradw_quintic)
dwdq_quintic_gpu = cuda.jit("float32(float32, float32, int64)", device=True)(dwdq_quintic)


if __name__ == '__main__':
    # sanity check for kernels; replication of Figure 2 from Smoothed Particle Hydrodynamics and Magnetohydrodynamics, Daniel J. Price
    # https://users.monash.edu.au/~dprice/ndspmhd/price-spmhd.pdf
    # f(x) and f'(x) only

    h = 1.0
    n_samples = 50

    r_vals = np.linspace(0.0, 3.0, n_samples)

    for dim in range(1,4):
        f_gauss = []
        fprime_gauss = []
        f_quintic = []
        fprime_quintic = []

        for r in r_vals:
            f_gauss.append(w_gauss(r, h, dim))
            fprime_gauss.append(dwdq_gauss(r, h, dim))

            f_quintic.append(w_quintic(r, h, dim))
            fprime_quintic.append(dwdq_quintic(r, h, dim))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.scatter(r_vals, f_gauss, c='#000000')
        ax1.scatter(r_vals, f_quintic, c='#000000', marker='*')

        ax1.scatter(r_vals, fprime_gauss, c='#ff0000')
        ax1.scatter(r_vals, fprime_quintic, c='#ff0000', marker='*')
        plt.show()