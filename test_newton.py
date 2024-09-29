import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
from numba import cuda, float32, int8, from_dtype

import pytest

#dim = 2
#norm_const = 1 / np.sqrt(np.pi)

#gauss_norms = [2/3, 10/(7*math.pi), 1/math.pi]
#quintic_norms = [1/120, 7/(478*math.pi), 1/(120*math.pi)]

#gauss_norm_2d = 10/(7*math.pi)
#sigma_gauss = 1/math.pi

gauss_norm_2d = 1/math.pi
gauss_norm_3d = math.pi / math.sqrt(math.pi)

quintic_norm_2d = 7/(478*math.pi)
quintic_norm_3d = 1/(120*math.pi)

eta_const = 1.0

M = 2 # star mass
R = 0.75 # star radius
particle_dim = 32
particle_mass = M / (particle_dim*particle_dim)

h_init = 0.1
#sigma = norms[dim-1]

# gpu stuff
threads = 16
tpb = (threads, threads)
bpg = ( int(particle_dim / threads), int(particle_dim / threads) )

kernel_radius = 3.0

def W(r, h, dim):

    if dim == 2:
        sigma = quintic_norm_2d
    else:
        sigma = quintic_norm_3d

    #print(h)
    h1 = 1. / h
    #print(h1)
    #exit()
    q = r * h1

    #print(q)
    #print(r.shape)
    #print(h1.shape)
    #exit()

    w = np.zeros(q.shape)

    #np.putmask(w, (q >= 0) & (q < 1), (1/4)*(2-q)**3 - (1-q)**3)
    #np.putmask(w, (q >= 1) & (q < 2), (1/4)*(2-q)**3)
    #np.putmask(w, q >= 2, 0.0)

    tmp3 = 3. - q
    tmp2 = 2. - q
    tmp1 = 1. - q

    val1 = tmp3 * tmp3 * tmp3 * tmp3 * tmp3
    val1 -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2

    val0 = tmp3 * tmp3 * tmp3 * tmp3 * tmp3
    val0 -= 6.0 * tmp2 * tmp2 * tmp2 * tmp2 * tmp2
    val0 += 15. * tmp1 * tmp1 * tmp1 * tmp1 * tmp1


    np.putmask(w, q >= 3.0, 0.0)
    np.putmask(w, (q >= 2) & (q < 3), tmp3 * tmp3 * tmp3 * tmp3 * tmp3)
    #print(w)
    np.putmask(w, (q >= 1) & (q < 2), val1)
    np.putmask(w, (q >= 0) & (q < 1), val0)

    #print(w)
    #exit()

    return (h1 ** dim) * sigma * w

'''
@cuda.jit('float32(float32, float32)', device=True)
def kernel_w_quintic(r, smoothing_length):
    # from phantom sph: https://github.com/danieljprice/phantom/blob/master/src/main/kernel_cubic.f90
    h1 = 1. / smoothing_length
    q = r * h1
    q2 = q*q

    fac = base_fac_cubic * h1**dim

    if (q < 1.):
        wkern  = 0.75*q2*q - 1.5*q2 + 1.
    elif (q < 2.):
        wkern  = -0.25*(q - 2.)**3
    else:
        wkern  = 0.

    return fac * wkern

@cuda.jit('float32(float32, float32)', device=True)
def dwdq_quintic(r, smoothing_length):
    # from phantom sph: https://github.com/danieljprice/phantom/blob/master/src/main/kernel_cubic.f90
    h1 = 1. / smoothing_length
    q = r * h1
    q2 = q*q

    fac = base_fac_cubic * h1**dim

    if (q < 1.):
        grkern = q*(2.25*q - 3.)
    elif (q < 2.):
        grkern = -0.75*(q - 2.)**2
    else:
        grkern = 0.

    return fac * grkern

@cuda.jit('float32(float32, float32)', device=True)
def kernel_grad_w_quintic(r, smoothing_length):
    # from pysph
    h1 = 1. / smoothing_length

    # compute the gradient.
    if (r > 1e-12):
        ##wdash = dwdq_cubic(r, smoothing_length)
        wdash = dwdq_cubic(r, smoothing_length)
        tmp = wdash * h1 / r
    else:
        tmp = 0.0

    return tmp
'''

@cuda.jit('float32(float32, float32, int64)', device=True)
def kernel_w_quintic(r, h, dim):
    # from phantom sph: https://github.com/danieljprice/phantom/blob/master/src/main/kernel_cubic.f90
    if dim == 2:
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

@cuda.jit('float32(float32, float32, int64)', device=True)
def kernel_grad_w_quintic(r, h, dim):
    # from phantom sph: https://github.com/danieljprice/phantom/blob/master/src/main/kernel_cubic.f90
    if dim == 2:
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
        grad_w = tmp3 * tmp3 * tmp3 * tmp3 * tmp3
    elif q >= 1:
        grad_w = val1
    else:
        grad_w = val0

    #if r > 1e-12:
    #    return (h1 ** dim) * sigma * grad_w * h1 / r
    #else:
    #    return 0.0
    return sigma * grad_w #* h1


@cuda.jit('void(float32[:,:,:], float32, float32[:,:,:])')
def calc_h_guesses(pos, kernel_r, h_guesses):
    i, j = cuda.grid(2)

    position = pos[i, j]
    dim = len(position)

    min_radius = np.inf
    max_radius = 0.0

    for i1 in range(pos.shape[0]):
        for j1 in range(pos.shape[1]):

            radius = 0.0
            for d in range(dim):
                radius += (position[d] - pos[i1, j1, d])**2

            radius = math.sqrt(radius)

            if radius > 0.0 and radius < min_radius:
                min_radius = radius
            if radius > max_radius:
                max_radius = radius

    h_guesses[i,j,0] = min_radius * (1 / kernel_r)
    h_guesses[i,j,1] = (max_radius + 1e-15) * (1 / kernel_r)


@cuda.jit('void(float32[:,:,:], float32, float32[:,:], float32[:,:])')
def calc_zeta(pos, mass, smoothing_lengths, zeta):
    i, j = cuda.grid(2)

    position = pos[i, j]
    dim = len(position)
    h = smoothing_lengths[i,j]

    rho = 0.0

    for i1 in range(pos.shape[0]):
        for j1 in range(pos.shape[1]):

            radius = 0.0
            for d in range(dim):
                radius += (position[d] - pos[i1, j1, d])**2

            radius = math.sqrt(radius)
            w = kernel_w_quintic(radius, h, dim)
            rho += particle_mass * w

    zeta[i,j] = particle_mass * ( (eta_const / h)**dim ) - rho

#@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:])')
#def calc_midpoint(x_low, x_high, midpoint):
#    i, j = cuda.grid(2)
#    midpoint[i, j] = 0.5 * (x_low[i,j] + x_high[i,j])

@cuda.jit('void(float32[:,:,:], float32[:,:])')
def calc_midpoint(x, midpoint):
    i, j = cuda.grid(2)
    midpoint[i, j] = 0.5 * (x[i,j,0] + x[i,j,1])

@cuda.jit('void(float32[:,:,:], float32[:,:], float32[:,:], float32[:,:])')
def bisect_update(x, x_midpt, y_low, y_midpt):
    i, j = cuda.grid(2)
    
    same_sign = y_low[i,j] * y_midpt[i, j] > 0

    if same_sign:
        x[i,j,0] = x_midpt[i,j]
    else:
        x[i,j,1] = x_midpt[i,j]

#@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:], float32[:,:], float32[:,:])')
#def bisect_update(x_low, x_high, x_midpt, y_low, y_midpt):
#    i, j = cuda.grid(2)
    
#    same_sign = y_low[i,j] * y_midpt[i, j] > 0

#    if same_sign:
#        x_low[i,j] = x_midpt[i,j]
#    else:
#        x_high[i,j] = x_midpt[i,j]




@cuda.jit('void(float32[:,:], float32, int64, boolean, float32[:,:])')
def kernel_gpu_test(radii, h, dim, is_grad, kernel_vals):
    i, j = cuda.grid(2)
    radius = radii[i, j]
    if is_grad:
        w = kernel_grad_w_quintic(radius, h, dim)
    else:
        w = kernel_w_quintic(radius, h, dim)
    kernel_vals[i,j] = w


def kernel_gauss( r, h, dim ):
    norm_const = 1 / np.sqrt(np.pi)

    h1 = 1.0 / h
    
    w = ( (h1 * norm_const) **dim )  * np.exp( -r**2 / h**2)
    
    return w

def kernel_grad_w_gauss(r, h, dim):
    norm_const = 1 / np.sqrt(np.pi)
    h1 = 1.0 / h
    return (-2 * h1**2) * ( (h1 * norm_const) **dim ) * np.exp( -r**2 / h**2)

def pairwise_separations_pmocz( ri, rj ):
	"""
	Get pairwise desprations between 2 sets of coordinates
	ri    is an M x 3 matrix of positions
	rj    is an N x 3 matrix of positions
	dx, dy, dz   are M x N matrices of separations
	"""
	
	M = ri.shape[0]
	N = rj.shape[0]
	
	# positions ri = (x,y,z)
	rix = ri[:,0].reshape((M,1))
	riy = ri[:,1].reshape((M,1))

    #if ri.shape[1] == 3:
	riz = ri[:,2].reshape((M,1)) if ri.shape[1] == 3 else None
	
	# other set of points positions rj = (x,y,z)
	rjx = rj[:,0].reshape((N,1))
	rjy = rj[:,1].reshape((N,1))
	rjz = rj[:,2].reshape((N,1)) if ri.shape[1] == 3 else None
	
	# matrices that store all pairwise particle separations: r_i - r_j
	dx = rix - rjx.T
	dy = riy - rjy.T
	dz = riz - rjz.T if ri.shape[1] == 3 else np.zeros(dx.shape)
	
	return dx, dy, dz

@pytest.fixture
def setup_data():
    np.random.seed(42)

    pos_2d = cuda.to_device( R*0.1 * np.random.randn(particle_dim, particle_dim, 2).astype('f4') )
    pos_3d = cuda.to_device( R*0.1 * np.random.randn(particle_dim, particle_dim, 3).astype('f4') )

    yield pos_2d, pos_3d


@pytest.mark.parametrize('spatial_dim', [2, 3])
def test_h_guesses(setup_data, spatial_dim):
    pos_2d, pos_3d = setup_data

    h_guesses = cuda.to_device(np.zeros((particle_dim, particle_dim, 2), dtype='f4'))
    zeta = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))

    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d

    pos_cpu = np.reshape(pos_gpu.copy_to_host(), (particle_dim*particle_dim, spatial_dim)).astype('f4')
    x_dist, y_dist, z_dist = pairwise_separations_pmocz( pos_cpu, pos_cpu )

    r_pmocz = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
    r_pmocz[r_pmocz == 0] = np.nan

    low_pmocz = np.nanmin(r_pmocz, axis=0) * (1 / kernel_radius)
    high_pmocz = np.nanmax(r_pmocz, axis=0) * (1 / kernel_radius)

    calc_h_guesses[bpg, tpb](pos_gpu, kernel_radius, h_guesses)
    h_guesses_cpu = np.reshape(h_guesses.copy_to_host(), (particle_dim*particle_dim, 2))
    low = h_guesses_cpu[:,0]
    high = h_guesses_cpu[:,1]

    assert all(np.isclose(low, low_pmocz))
    assert all(np.isclose(high, high_pmocz))


@pytest.mark.parametrize('spatial_dim, k', [
    (2, 'w'),
    (3, 'w'),
    #(2, 'grad_w'),
    #(3, 'grad_w'),
])
def test_kernel(setup_data, spatial_dim, k):
    pos_2d, pos_3d = setup_data

    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d
    gauss_kernel_func = kernel_grad_w_gauss if k == 'grad_w' else kernel_gauss
    is_grad = k == 'grad_w'

    pos_cpu = np.reshape(pos_gpu.copy_to_host(), (particle_dim*particle_dim, spatial_dim)).astype('f4')
    x_dist, y_dist, z_dist = pairwise_separations_pmocz( pos_cpu, pos_cpu )
    pair_dists = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2).astype('f4')
    r_pmocz = pair_dists[:particle_dim, :particle_dim].copy()

    kernel_vals = cuda.to_device(np.zeros(r_pmocz.shape).astype('f4'))
    
    gauss_res = gauss_kernel_func(r_pmocz, h_init, spatial_dim)

    assert not np.isnan(gauss_res).any()

    radii = cuda.to_device(r_pmocz)

    kernel_gpu_test[bpg, tpb](radii, h_init, spatial_dim, is_grad, kernel_vals)

    quintic_res = kernel_vals.copy_to_host()

    assert not np.isnan(quintic_res).any()

    diff = np.abs(gauss_res-quintic_res)

    zero_vals = np.zeros(gauss_res.shape)
    np.putmask(zero_vals, (gauss_res == 0) | (quintic_res == 0), diff)

    assert np.all(zero_vals < 0.02)
    #print(np.max(zero_vals))

    #print(gauss_res)
    #print(quintic_res)

    max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
    
    assert diff[max_diff_idx] / max(gauss_res[max_diff_idx], quintic_res[max_diff_idx]) < 0.04



@pytest.mark.parametrize('spatial_dim', [2, 3])
def test_zeta(setup_data, spatial_dim):
    n_samples = 50

    pos_2d, pos_3d = setup_data

    h_guesses = cuda.to_device(np.zeros((particle_dim, particle_dim, 2), dtype='f4'))
    zeta = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))

    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d
    calc_h_guesses[bpg, tpb](pos_gpu, kernel_radius, h_guesses)

    calc_zeta[bpg, tpb](pos_gpu, particle_mass, h_guesses[:,:,0], zeta)
    zeta_low = zeta.copy_to_host()
    assert np.all(zeta_low > 0.0)

    calc_zeta[bpg, tpb](pos_gpu, particle_mass, h_guesses[:,:,1], zeta)
    zeta_high = zeta.copy_to_host()
    assert np.all(zeta_high < 0.0)

    all_hvals = np.linspace(h_guesses[:,:,0], h_guesses[:,:,1], n_samples).astype('f4')

    sign_test_vals = np.zeros((particle_dim, particle_dim, n_samples), dtype='f4')

    for i in range(n_samples):
        h_iter = cuda.to_device(np.ascontiguousarray(all_hvals[i,:,:]))
        calc_zeta[bpg, tpb](pos_gpu, particle_mass, h_iter, zeta)
        zeta_cpu = zeta.copy_to_host()
        sign_test_vals[:,:,i] = zeta_cpu


    asign = np.sign(sign_test_vals)
    sign_change = ((np.roll(asign, 1) - asign) != 0).astype(int)[:,:,1:]

    sign_change_sum = np.sum(sign_change, axis=-1)

    assert np.all(sign_change_sum == 1)



@pytest.mark.parametrize('spatial_dim', [2, 3])
def test_newton_method(setup_data, spatial_dim):
    pos_2d, pos_3d = setup_data
    n_samples = 50
    n_iter = 30

    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d
    x = cuda.to_device(np.zeros((particle_dim, particle_dim, 2), dtype='f4'))
    calc_h_guesses[bpg, tpb](pos_gpu, kernel_radius, x)

    y_low = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
    y_high = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
    
    x_midpt = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
    y_midpt = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))


    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d
    calc_h_guesses[bpg, tpb](pos_gpu, kernel_radius, x)
    
    calc_zeta[bpg, tpb](pos_gpu, particle_mass, x[:,:,0], y_low)
    calc_zeta[bpg, tpb](pos_gpu, particle_mass, x[:,:,1], y_high)


    for i in range(n_iter):
        calc_midpoint[bpg, tpb](x, x_midpt)
        calc_zeta[bpg, tpb](pos_gpu, particle_mass, x_midpt, y_midpt)
        bisect_update[bpg, tpb](x, x_midpt, y_low, y_midpt)

    midpt_y_cpu = np.abs(y_midpt.copy_to_host())

    assert np.all(midpt_y_cpu < h_init * 5e-2)




if __name__ == '__main__':
    np.random.seed(42)

    pos_2d = cuda.to_device( R*0.0001 * np.random.randn(particle_dim, particle_dim, 2).astype('f4') )
    pos_3d = cuda.to_device( R*0.0001 * np.random.randn(particle_dim, particle_dim, 3).astype('f4') )

    test_newton_method(pos_2d, pos_3d, 2)


    '''
    def dwdq(rij, h, dim):

        if dim == 2:
            fac = (1.0 / math.pi) * 7.0 / 478.0
        else:
            fac = (1.0 / math.pi) * 1.0 / 120.0

        h1 = 1. / h
        q = rij * h1

        # get the kernel normalizing factor
        if dim == 1:
            fac = fac * h1
        elif dim == 2:
            fac = fac * h1 * h1
        elif dim == 3:
            fac = fac * h1 * h1 * h1

        tmp3 = 3. - q
        tmp2 = 2. - q
        tmp1 = 1. - q

        # compute the gradient
        #if (rij > 1e-12):
        if (q > 3.0):
            val = 0.0

        elif (q > 2.0):
            val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3

        elif (q > 1.0):
            val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
            val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2
        else:
            val = -5.0 * tmp3 * tmp3 * tmp3 * tmp3
            val += 30.0 * tmp2 * tmp2 * tmp2 * tmp2
            val -= 75.0 * tmp1 * tmp1 * tmp1 * tmp1
        #else:
        #    val = 0.0

        return val * fac

    def gradW_gauss( r, h, dim ):
        """
        Gradient of the Gausssian Smoothing kernel (3D)
        x     is a vector/matrix of x positions
        y     is a vector/matrix of y positions
        z     is a vector/matrix of z positions
        h     is the smoothing length
        wx, wy, wz     is the evaluated gradient
        """
        norm_const = 1 / np.sqrt(np.pi)
                
        h1 = 1.0 / h

        n = (-2 * h1**2) * ( (h1 * norm_const) **dim ) * np.exp( -r**2 / h**2)
        #n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
        #wx = n * x
        #wy = n * y
        #wz = n * z
        
        return n#wx, wy#, wz


    pos = R*0.0001 * np.random.randn(particle_dim, particle_dim, 3).astype('f4')

    pos = np.reshape(pos, (particle_dim*particle_dim, 3))

    x_dist, y_dist, z_dist = pairwise_separations_pmocz( pos, pos )
    pair_dists = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2).astype('f4')
    r_pmocz = pair_dists[:particle_dim, :particle_dim].copy()

    print(gradW_gauss(r_pmocz, h_init, 3))

    #print(dwdq(r_pmocz, h_init, 3))

    res = np.zeros(r_pmocz.shape)

    for i in range(r_pmocz.shape[0]):
        for j in range(r_pmocz.shape[1]):
            res[i,j] = dwdq(r_pmocz[i,j], h_init, 3)
    print(res)
'''