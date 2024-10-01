import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
from numba import cuda, float32, int8, from_dtype
from kernels import w_gauss, dwdq_gauss, w_quintic_gpu, dwdq_quintic_gpu
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths

import pytest

M = 2 # star mass
R = 0.75 # star radius
particle_dim = 32
particle_mass = M / (particle_dim*particle_dim)

kernel_radius = 3.0


h_init = 0.1
#sigma = norms[dim-1]

# gpu stuff
threads = 16
tpb = (threads, threads)
bpg = ( int(particle_dim / threads), int(particle_dim / threads) )


@cuda.jit('void(float32[:,:], float32, int64, boolean, float32[:,:])')
def kernel_gpu_test(radii, h, dim, is_grad, kernel_vals):
    i, j = cuda.grid(2)
    radius = radii[i, j]
    if is_grad:
        w = dwdq_quintic_gpu(radius, h, dim)
    else:
        w = w_quintic_gpu(radius, h, dim)
    kernel_vals[i,j] = w


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

    pos_2d = cuda.to_device( R * np.random.randn(particle_dim, particle_dim, 2).astype('f4') )
    pos_3d = cuda.to_device( R * np.random.randn(particle_dim, particle_dim, 3).astype('f4') )

    yield pos_2d, pos_3d


@pytest.mark.parametrize('spatial_dim, k', [
    (2, 'w'),
    (3, 'w'),
    (2, 'grad_w'),
    (3, 'grad_w'),
])
def test_kernel(setup_data, spatial_dim, k):
    pos_2d, pos_3d = setup_data

    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d
    #gauss_kernel_func = kernel_grad_w_gauss if k == 'grad_w' else kernel_gauss
    gauss_kernel_func = dwdq_gauss if k == 'grad_w' else w_gauss
    is_grad = k == 'grad_w'

    pos_cpu = np.reshape(pos_gpu.copy_to_host(), (particle_dim*particle_dim, spatial_dim)).astype('f4')
    x_dist, y_dist, z_dist = pairwise_separations_pmocz( pos_cpu, pos_cpu )
    pair_dists = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2).astype('f4')
    r_pmocz = pair_dists[:particle_dim, :particle_dim].copy()

    kernel_vals = cuda.to_device(np.zeros(r_pmocz.shape).astype('f4'))
    
    gauss_res = gauss_kernel_func(r_pmocz, h_init, spatial_dim)

    if is_grad:
        np.fill_diagonal(gauss_res, 0.0)

    assert not np.isnan(gauss_res).any()

    radii = cuda.to_device(r_pmocz)

    kernel_gpu_test[bpg, tpb](radii, h_init, spatial_dim, is_grad, kernel_vals)

    quintic_res = kernel_vals.copy_to_host()

    if is_grad:
        np.fill_diagonal(quintic_res, 0.0)

    assert not np.isnan(quintic_res).any()

    diff = np.abs(gauss_res-quintic_res)

    zero_vals = np.zeros(gauss_res.shape)
    np.putmask(zero_vals, (gauss_res == 0) | (quintic_res == 0), diff)

    assert np.all(zero_vals < 0.5)

    max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
    
    assert diff[max_diff_idx] / max(gauss_res[max_diff_idx], quintic_res[max_diff_idx]) < 0.05

@pytest.mark.parametrize('spatial_dim', [2, 3])
def test_h_guesses(setup_data, spatial_dim):
    pos_2d, pos_3d = setup_data

    h_guesses = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
    zeta = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))

    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d

    pos_cpu = np.reshape(pos_gpu.copy_to_host(), (particle_dim*particle_dim, spatial_dim)).astype('f4')
    x_dist, y_dist, z_dist = pairwise_separations_pmocz( pos_cpu, pos_cpu )

    r_pmocz = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
    r_pmocz[r_pmocz == 0] = np.nan

    low_pmocz = np.nanmin(r_pmocz, axis=0) * (1 / kernel_radius)
    high_pmocz = np.nanmax(r_pmocz, axis=0) * (1 / kernel_radius)

    calc_h_guesses[bpg, tpb](pos_gpu, kernel_radius, h_guesses)
    h_guesses_cpu = np.reshape(h_guesses.copy_to_host(), (particle_dim*particle_dim, 3))
    low = h_guesses_cpu[:,0]
    high = h_guesses_cpu[:,2]

    assert all(np.isclose(low, low_pmocz))
    assert all(np.isclose(high, high_pmocz))


@pytest.mark.parametrize('spatial_dim', [2, 3])
def test_zeta(setup_data, spatial_dim):
    n_samples = 50

    pos_2d, pos_3d = setup_data

    h_guesses = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
    zeta = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))

    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d
    calc_h_guesses[bpg, tpb](pos_gpu, kernel_radius, h_guesses)

    calc_zeta[bpg, tpb](pos_gpu, particle_mass, h_guesses[:,:,0], zeta)
    zeta_low = zeta.copy_to_host()
    assert np.all(zeta_low > 0.0)

    calc_zeta[bpg, tpb](pos_gpu, particle_mass, h_guesses[:,:,2], zeta)
    zeta_high = zeta.copy_to_host()
    assert np.all(zeta_high < 0.0)

    all_hvals = np.linspace(h_guesses[:,:,0], h_guesses[:,:,2], n_samples).astype('f4')

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
    x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
    y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
    calc_h_guesses[bpg, tpb](pos_gpu, kernel_radius, x)


    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d
    
    get_new_smoothing_lengths(pos_gpu, x, y, particle_mass, kernel_radius, tpb, bpg, n_iter=30)

    midpt_y_cpu = np.abs(y.copy_to_host()[:,:,1])

    #pos_cpu = pos_gpu.copy_to_host()
    #new_smoothing_length = x.copy_to_host()[:,:,1]
    #for i in range(particle_dim):
    #    for j in range(particle_dim):
    #        print('-----')
    #        print(new_smoothing_length[i,j])
    #        print(np.linalg.norm(pos_cpu[i,j]))



    assert not np.any(midpt_y_cpu == 0.0)
    assert np.all(midpt_y_cpu < h_init * 5e-2)
