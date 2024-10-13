import math
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import cuda
from scipy.special import gamma
from PIL import Image

from astro_core import calc_dv_polytrope, calc_density, leapfrog_update_nd, calc_mean_nd
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths
from pmocz_functions import plot_frame, get_img


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
    #(3, 32, lmbda_3d)
]

for spatial_dim, particle_dim, lmbda in configs:

    threads = 16
    tpb = (threads, threads)
    bpg = ( int(particle_dim / threads), int(particle_dim / threads) )


    N = particle_dim*particle_dim    # Number of particles

    smoothing_length = np.zeros((particle_dim, particle_dim)) + h_init
    smoothing_length = cuda.to_device(smoothing_length.astype('f4'))

    #init_pos = (R * 4/3) * np.random.randn(particle_dim, particle_dim, 2, spatial_dim).astype('f4')
    init_pos = (R * 4/3) * np.random.randn(particle_dim, particle_dim, spatial_dim).astype('f4')
    d_pos = cuda.to_device(init_pos)

    pos_i = cuda.to_device(np.zeros(d_pos.shape, dtype='f4'))

    d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
    #d_vel = cuda.to_device(np.zeros((particle_dim, particle_dim, 2, spatial_dim), dtype='f4'))
    d_vel = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))
    vel_i = cuda.to_device(np.zeros(d_vel.shape, dtype='f4'))
    vel_mean =cuda.to_device( np.zeros(d_vel.shape, dtype='f4'))

    d_dV = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))

    particle_mass = M/N
    steps = int(tEnd/dt)

    all_pos = np.zeros((N, spatial_dim, steps), dtype='f4')
    all_rho = np.zeros((N, steps), dtype='f4')

    x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
    y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))

    imgs = []

    # calc_dv_polytrope[bpg, tpb](d_pos[:,:,0,:], d_vel[:,:,0,:], particle_mass, smoothing_length, eq_state_const, polytropic_idx, lmbda, d_rho, d_dV)


    for i in range(1, steps):
        print(i)

        
        pos_i[:] = d_pos
        vel_i[:] = d_vel

        #update_param(d_pos, d_pos, d_vel, dt*0.5, tpb, bpg)
        #update_param(d_vel, d_vel, d_dV, dt*0.5, tpb, bpg)
        leapfrog_update_nd[bpg, tpb](d_pos, d_pos, d_vel, dt*0.5)
        leapfrog_update_nd[bpg, tpb](d_vel, d_vel, d_dV, dt*0.5)

        get_new_smoothing_lengths(d_pos, x, y, particle_mass, 3.0, tpb, bpg, n_iter=15)
        smoothing_length = x[:,:,1]
        x_cpu = x.copy_to_host()
        delta_ratio = np.max(np.abs(x_cpu[:,:,2] - x_cpu[:,:,0]) / h_init)
        assert np.all(delta_ratio < 0.02 * h_init)

        calc_density[bpg, tpb](d_pos, particle_mass, smoothing_length, d_rho)
        calc_dv_polytrope[bpg, tpb](d_pos, d_vel, particle_mass, smoothing_length, eq_state_const, polytropic_idx, lmbda, d_rho, d_dV)

        #update_param(vel_i, d_vel, d_dV, dt, tpb, bpg)
        leapfrog_update_nd[bpg, tpb](vel_i, d_vel, d_dV, dt)

        #get_mean_velocity(vel_i, d_vel, vel_mean, tpb, bpg)
        calc_mean_nd[bpg, tpb](vel_i, d_vel, vel_mean)
        #update_param(pos_i, d_pos, vel_mean, dt, tpb, bpg)
        leapfrog_update_nd[bpg, tpb](pos_i, d_pos, vel_mean, dt)

        all_pos[:,:,i] = (d_pos.copy_to_host()).reshape((N, spatial_dim))
        all_rho[:,i] = (d_rho.copy_to_host()).reshape((N,))

        imgs.append(get_img(all_pos[:,:,i], all_rho[:,i], R, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda))

    gif_path = os.path.join(os.path.dirname(__file__), f'figures/04_relax_polytrope/toystar_{spatial_dim}d.gif')
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=20, loop=0)

    png_path = os.path.join(os.path.dirname(__file__), f'figures/04_relax_polytrope/toystar_{spatial_dim}d.png')
    fig = plot_frame(all_pos[:,:,-1], all_rho[:,-1], R, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda, colormap='jet')
    plt.savefig(png_path)
    plt.close(fig)

    #np_path = os.path.join(os.path.dirname(__file__), f'data/toystar_pos_{spatial_dim}d.npy')

    #with open(np_path,'wb') as f:
    #    np.save(f, all_pos)