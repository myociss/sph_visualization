import math
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import cuda
from scipy.special import gamma
from gpu_core import calc_dv_toystar, calc_density, update_pos_vel_halfstep
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths
from pmocz_functions import plot_frame

# demonstrates formation of a toy star integrated using a quintic spline kernel and adaptive smoothing length in 2D and 3D

R = 0.75 # star radius
M = 2 # star mass
polytropic_idx = 3/2


h_init = 0.1 * (R * 4/3) #0.1 * (R * 4/3)    # smoothing length
eq_state_const = 0.1 * (R * 4/3)   # equation of state constant
viscosity = 1      # damping

lmbda_2d = 2*eq_state_const*np.pi**(-1/polytropic_idx) * ( ( (M*(1+polytropic_idx)) / (R**2) )**(1 + 1/polytropic_idx) ) / M
lmbda_3d = 2*eq_state_const*(1+polytropic_idx)*np.pi**(-3/(2*polytropic_idx)) * (M*gamma(5/2+polytropic_idx)/R**3/gamma(1+polytropic_idx))**(1/polytropic_idx) / R**2

t         = 0      # current time of the simulation
tEnd      = 12     # time at which simulation ends
#dt        = 0.01   # timestep
dt = 0.005#0.01

configs = [
    (2, 16, lmbda_2d),
    (3, 32, lmbda_3d)
]

for spatial_dim, particle_dim, lmbda in configs:

    #particle_dim = 32
    threads = 16
    tpb = (threads, threads)
    bpg = ( int(particle_dim / threads), int(particle_dim / threads) )


    N         = particle_dim*particle_dim    # Number of particles

    smoothing_length = np.zeros((particle_dim, particle_dim)) + h_init
    smoothing_length = cuda.to_device(smoothing_length.astype('f4'))

    #spatial_dim = 3

    init_pos = (R * 4/3) * np.random.randn(particle_dim, particle_dim, 2, spatial_dim).astype('f4')
    d_pos = cuda.to_device(init_pos)

    d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
    d_vel = cuda.to_device(np.zeros((particle_dim, particle_dim, 2, spatial_dim), dtype='f4'))
    d_dV = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))

    particle_mass = M/N

    #lmbda_2d = 2*eq_state_const*np.pi**(-1/polytropic_idx) * ( ( (M*(1+polytropic_idx)) / (R**2) )**(1 + 1/polytropic_idx) ) / M
    #lmbda_3d = 2*eq_state_const*(1+polytropic_idx)*np.pi**(-3/(2*polytropic_idx)) * (M*gamma(5/2+polytropic_idx)/R**3/gamma(1+polytropic_idx))**(1/polytropic_idx) / R**2

    #lmbda = lmbda_2d if spatial_dim == 2 else lmbda_3d

    #print(lmbda)

    steps = int(tEnd/dt)

    all_pos = np.zeros((N, spatial_dim, steps), dtype='f4')

    all_pos[:,:, 0] = init_pos[:,:,0,:].reshape((N, spatial_dim))

    x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
    y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))

    for i in range(1, steps):
        print(i)

        update_pos_vel_halfstep[bpg, tpb](d_pos, d_vel, d_dV, dt)
        calc_density[bpg, tpb](d_pos[:,:,0,:], particle_mass, smoothing_length, d_rho)

        get_new_smoothing_lengths(d_pos[:,:,0,:], x, y, particle_mass, 3.0, tpb, bpg, n_iter=30)
        smoothing_length = x[:,:,1]
        smoothing_length_y_cpu = y[:,:,1].copy_to_host()
        assert np.all(np.abs(smoothing_length_y_cpu) < 0.001 * R)
        

        calc_dv_toystar[bpg, tpb](d_pos[:,:,0,:], d_vel[:,:,0,:], particle_mass, smoothing_length, eq_state_const, polytropic_idx, lmbda, viscosity, d_rho, d_dV)

        update_pos_vel_halfstep[bpg, tpb](d_pos, d_vel, d_dV, dt)

        all_pos[:,:, i] = (d_pos.copy_to_host()[:,:,0,:]).reshape((N, spatial_dim))


    density = d_rho.copy_to_host()

    pos = all_pos[:,:,-1]
    path = os.path.join(os.path.dirname(__file__), f'figures/01_toystar/toystar_{spatial_dim}d.png')
    plot_frame(pos, density, R, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda, path)
