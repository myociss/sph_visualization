import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma
from numba import cuda, float32, int8, from_dtype
from kernels import w_gauss, dwdq_gauss, w_quintic_gpu, dwdq_quintic_gpu
from astro_core import calc_density, calc_dv_toystar
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths
from pmocz_functions import pairwise_separations, density, dv
import pytest

M = 2 # star mass
R = 0.75 # star radius

#R = 7000000 # white dwarf radius
#solar_mass = 1.989e30
#M=0.6*solar_mass # white dwarf mass
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


@pytest.fixture
def setup_data():
    np.random.seed(42)

    pos_3d = cuda.to_device( R * (4/3) * np.random.randn(particle_dim, particle_dim, 3).astype('f4') )
    pos_2d = cuda.to_device( R * (4/3) * np.random.randn(particle_dim, particle_dim, 2).astype('f4') )

    yield pos_2d, pos_3d


@pytest.mark.parametrize('spatial_dim', [2, 3])
def test_rho(setup_data, spatial_dim):
    pos_2d, pos_3d = setup_data
    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d

    pos_cpu = np.reshape(pos_gpu.copy_to_host(), (particle_dim*particle_dim, spatial_dim)).astype('f4')

    particle_masses = np.zeros((particle_dim, particle_dim)) + particle_mass
    particle_masses = cuda.to_device(particle_masses.astype('f4'))
    rho_pmocz = density(pos_cpu, h_init, particle_mass, spatial_dim)

    smoothing_lengths = np.zeros((particle_dim, particle_dim)) + h_init
    smoothing_lengths = cuda.to_device(smoothing_lengths.astype('f4'))
    rho = cuda.to_device(np.zeros((particle_dim, particle_dim)).astype('f4'))
    calc_density[bpg, tpb](pos_gpu, particle_masses, smoothing_lengths, rho)

    rho_cpu = np.reshape(rho.copy_to_host(), (particle_dim*particle_dim,1))

    ratio = np.abs(rho_cpu - rho_pmocz) / np.minimum(rho_cpu, rho_pmocz)    

    assert np.all(ratio < 0.06)


@pytest.mark.parametrize('spatial_dim', [2, 3])
def test_dv(setup_data, spatial_dim):
    pos_2d, pos_3d = setup_data
    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d

    eq_state_const = 0.1 * (R * 4/3)
    polytropic_idx = 1
    viscosity = 1

    velocity = R * 0.2 * np.random.randn(particle_dim, particle_dim, spatial_dim).astype('f4')

    lmbda_2d = 2*eq_state_const*np.pi**(-1/polytropic_idx) * ( ( (M*(1+polytropic_idx)) / (R**2) )**(1 + 1/polytropic_idx) ) / M
    lmbda_3d = 2*eq_state_const*(1+polytropic_idx)*np.pi**(-3/(2*polytropic_idx)) * (M*gamma(5/2+polytropic_idx)/R**3/gamma(1+polytropic_idx))**(1/polytropic_idx) / R**2

    lmbda = lmbda_2d if spatial_dim == 2 else lmbda_3d

    pos_cpu = np.reshape(pos_gpu.copy_to_host(), (particle_dim*particle_dim, spatial_dim)).astype('f4')

    particle_masses = np.zeros((particle_dim, particle_dim)) + particle_mass
    particle_masses = cuda.to_device(particle_masses.astype('f4'))

    acc_pmocz = dv(pos_cpu, np.reshape(velocity, (particle_dim*particle_dim,spatial_dim)), h_init, eq_state_const, polytropic_idx, lmbda, viscosity, particle_mass, spatial_dim)

    acc = cuda.to_device( np.random.randn(particle_dim, particle_dim, spatial_dim).astype('f4') )# make sure setting dV[i,j,d] = 0 works
    vel = cuda.to_device(velocity)

    smoothing_lengths = np.zeros((particle_dim, particle_dim)) + h_init
    smoothing_lengths = cuda.to_device(smoothing_lengths.astype('f4'))

    rho = cuda.to_device(np.zeros((particle_dim, particle_dim)).astype('f4'))
    calc_density[bpg, tpb](pos_gpu, particle_masses, smoothing_lengths, rho)

    calc_dv_toystar[bpg, tpb](pos_gpu, vel, particle_masses, smoothing_lengths, eq_state_const, polytropic_idx, lmbda, viscosity, rho, acc)
    acc_cpu = np.reshape(acc.copy_to_host(), acc_pmocz.shape)

    assert np.all(np.abs(acc_pmocz - acc_cpu) < 0.2)
    

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
    x_dist, y_dist, z_dist = pairwise_separations( pos_cpu, pos_cpu )
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
    mask = cuda.to_device(np.ones((particle_dim, particle_dim), dtype='f4'))

    pos_cpu = np.reshape(pos_gpu.copy_to_host(), (particle_dim*particle_dim, spatial_dim)).astype('f4')
    x_dist, y_dist, z_dist = pairwise_separations( pos_cpu, pos_cpu )

    r_pmocz = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
    r_pmocz[r_pmocz == 0] = np.nan

    low_pmocz = np.nanmin(r_pmocz, axis=0) * (1 / kernel_radius)
    high_pmocz = np.nanmax(r_pmocz, axis=0) * (1 / kernel_radius)

    calc_h_guesses[bpg, tpb](pos_gpu, mask, kernel_radius, h_guesses)
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
    mask = cuda.to_device(np.ones((particle_dim, particle_dim), dtype='f4'))

    particle_masses = np.zeros((particle_dim, particle_dim)) + particle_mass
    particle_masses = cuda.to_device(particle_masses.astype('f4'))

    calc_h_guesses[bpg, tpb](pos_gpu, mask, kernel_radius, h_guesses)

    calc_zeta[bpg, tpb](pos_gpu, mask, particle_masses, h_guesses[:,:,0], zeta)
    zeta_low = zeta.copy_to_host()
    assert np.all(zeta_low > 0.0)

    calc_zeta[bpg, tpb](pos_gpu, mask, particle_masses, h_guesses[:,:,2], zeta)
    zeta_high = zeta.copy_to_host()
    assert np.all(zeta_high < 0.0)

    all_hvals = np.linspace(h_guesses[:,:,0], h_guesses[:,:,2], n_samples).astype('f4')

    sign_test_vals = np.zeros((particle_dim, particle_dim, n_samples), dtype='f4')

    for i in range(n_samples):
        h_iter = cuda.to_device(np.ascontiguousarray(all_hvals[i,:,:]))
        calc_zeta[bpg, tpb](pos_gpu, mask, particle_masses, h_iter, zeta)
        zeta_cpu = zeta.copy_to_host()
        sign_test_vals[:,:,i] = zeta_cpu

    asign = np.sign(sign_test_vals)
    sign_change = ((np.roll(asign, 1) - asign) != 0).astype(int)[:,:,1:]

    sign_change_sum = np.sum(sign_change, axis=-1)

    assert np.all(sign_change_sum == 1)



@pytest.mark.parametrize('spatial_dim', [2, 3])
def test_bisection_method(setup_data, spatial_dim):
    pos_2d, pos_3d = setup_data
    n_samples = 50
    n_iter = 30

    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d
    x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
    y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))

    mask = cuda.to_device(np.ones((particle_dim, particle_dim), dtype='f4'))
    particle_masses = np.zeros((particle_dim, particle_dim)) + particle_mass
    particle_masses = cuda.to_device(particle_masses.astype('f4'))

    calc_h_guesses[bpg, tpb](pos_gpu, mask, kernel_radius, x)

    all_hvals = np.linspace(x[:,:,0], x[:,:,2], n_samples).astype('f4')

    pos_gpu = pos_2d if spatial_dim == 2 else pos_3d
    
    get_new_smoothing_lengths(pos_gpu, x, y, particle_masses, kernel_radius, tpb, bpg, mask, n_iter=30)

    x_cpu = x.copy_to_host()

    delta_ratio = np.max(np.abs(x_cpu[:,:,2] - x_cpu[:,:,0]) / R)
    assert np.all(delta_ratio < 0.001 * R) # error as a percentage of radius

    '''
    sign_test_vals = np.zeros((particle_dim, particle_dim, n_samples), dtype='f4')

    zeta = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))

    for i in range(n_samples):
        h_iter = cuda.to_device(np.ascontiguousarray(all_hvals[i,:,:]))
        calc_zeta[bpg, tpb](pos_gpu, mask, particle_mass, h_iter, zeta)
        zeta_cpu = zeta.copy_to_host()
        sign_test_vals[:,:,i] = zeta_cpu

    problem_idx = np.unravel_index(np.argmax(np.abs(midpt_y_cpu)), midpt_y_cpu.shape)
    print(problem_idx)
    print(midpt_y_cpu[problem_idx])
    print(midpt_x_cpu[problem_idx])
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(all_hvals[:,problem_idx[0],problem_idx[1]], sign_test_vals[problem_idx])
    ax1.scatter([midpt_x_cpu[problem_idx]], [midpt_y_cpu[problem_idx]])
    plt.show()
    '''
