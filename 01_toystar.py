import math
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import cuda
from scipy.special import gamma
from PIL import Image

from astro_core import calc_dv_toystar, calc_density, update_pos_vel_halfstep
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths
from pmocz_functions import plot_frame, get_img


# demonstrates formation of a toy star integrated using a quintic spline kernel and adaptive smoothing length in 2D and 3D

R = 0.75 # star radius
M = 2 # star mass
polytropic_idx = 3/2


h_init = 0.1 * (R * 4/3)    # smoothing length
eq_state_const = 0.1 * (R * 4/3) # equation of state constant
viscosity = 1 # damping

lmbda_2d = 2*eq_state_const*np.pi**(-1/polytropic_idx) * ( ( (M*(1+polytropic_idx)) / (R**2) )**(1 + 1/polytropic_idx) ) / M
lmbda_3d = 2*eq_state_const*(1+polytropic_idx)*np.pi**(-3/(2*polytropic_idx)) * (M*gamma(5/2+polytropic_idx)/R**3/gamma(1+polytropic_idx))**(1/polytropic_idx) / R**2

tEnd = 12
dt = 0.005

configs = [
    (2, 16, lmbda_2d),
    (3, 32, lmbda_3d)
]

for spatial_dim, particle_dim, lmbda in configs:

    threads = 16
    tpb = (threads, threads)
    bpg = ( int(particle_dim / threads), int(particle_dim / threads) )


    N = particle_dim*particle_dim    # Number of particles

    smoothing_length = np.zeros((particle_dim, particle_dim)) #+ h_init
    smoothing_length = cuda.to_device(smoothing_length.astype('f4'))

    init_pos = (R * 4/3) * np.random.randn(particle_dim, particle_dim, 2, spatial_dim).astype('f4')
    d_pos = cuda.to_device(init_pos)

    d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
    d_vel = cuda.to_device(np.zeros((particle_dim, particle_dim, 2, spatial_dim), dtype='f4'))
    d_dV = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))

    particle_mass = M/N

    steps = int(tEnd/dt)

    all_pos = np.zeros((N, spatial_dim, steps), dtype='f4')
    all_rho = np.zeros((N, steps), dtype='f4')

    x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
    y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))

    imgs = []

    for i in range(1, steps):
        print(i)

        update_pos_vel_halfstep[bpg, tpb](d_pos, d_vel, d_dV, dt)
        calc_density[bpg, tpb](d_pos[:,:,0,:], particle_mass, smoothing_length, d_rho)

        get_new_smoothing_lengths(d_pos[:,:,0,:], x, y, particle_mass, 3.0, tpb, bpg, n_iter=15)
        smoothing_length = x[:,:,1]
        #smoothing_length_y_cpu = y[:,:,1].copy_to_host()
        #assert np.all(np.abs(smoothing_length_y_cpu) < 0.001 * R)

        x_cpu = x.copy_to_host()
        delta_ratio = np.max(np.abs(x_cpu[:,:,2] - x_cpu[:,:,0]) / h_init)
        assert np.all(delta_ratio < 0.02 * h_init)
        

        calc_dv_toystar[bpg, tpb](d_pos[:,:,0,:], d_vel[:,:,0,:], particle_mass, smoothing_length, eq_state_const, polytropic_idx, lmbda, viscosity, d_rho, d_dV)

        update_pos_vel_halfstep[bpg, tpb](d_pos, d_vel, d_dV, dt)

        all_pos[:,:,i] = (d_pos.copy_to_host()[:,:,0,:]).reshape((N, spatial_dim))
        all_rho[:,i] = (d_rho.copy_to_host()).reshape((N,))

        imgs.append(get_img(all_pos[:,:,i], all_rho[:,i], R, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda))

    gif_path = os.path.join(os.path.dirname(__file__), f'figures/01_toystar/toystar_{spatial_dim}d.gif')
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=20, loop=0)

    png_path = os.path.join(os.path.dirname(__file__), f'figures/01_toystar/toystar_{spatial_dim}d.png')
    fig = plot_frame(all_pos[:,:,-1], all_rho[:,-1], R, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda, colormap='jet')
    plt.savefig(png_path)
    plt.close(fig)

    np_path = os.path.join(os.path.dirname(__file__), f'data/toystar_pos_{spatial_dim}d.npy')

    with open(np_path,'wb') as f:
        np.save(f, all_pos)