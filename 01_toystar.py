import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from gpu_core import calc_dV_toystar, calc_density, update_pos_vel_halfstep
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths
from numba import cuda

# cuda stuff
particle_dim = 32
threads = 16
tpb = (threads, threads)
bpg = ( int(particle_dim / threads), int(particle_dim / threads) )
N = particle_dim * particle_dim

# astrophysics stuff
R = 7000000 # white dwarf radius
solar_mass = 1.989e30
M=0.6*solar_mass # white dwarf mass

eq_state_const = 0.1 * (R * 4/3)   # equation of state constant
polytropic_idx = 3/2 # polytropic index
viscosity = 1 # damping

particle_mass = M/N

lmbda_2d = 2*eq_state_const*np.pi**(-1/polytropic_idx) * ( ( (M*(1+polytropic_idx)) / (R**2) )**(1 + 1/polytropic_idx) ) / M
lmbda_3d = 2*eq_state_const*(1+polytropic_idx)*np.pi**(-3/(2*polytropic_idx)) * (M*gamma(5/2+polytropic_idx)/R**3/gamma(1+polytropic_idx))**(1/polytropic_idx) / R**2


# simulation stuff
t = 0
tEnd = 12
dt = 0.005

kernel_radius = 3.0

for d in (2,3):
    init_pos = (R * 4/3) * np.random.randn(particle_dim, particle_dim, 2, d).astype('f4')

    d_pos = cuda.to_device(init_pos)
    d_vel = cuda.to_device(np.zeros((particle_dim, particle_dim, 2, d), dtype='f4'))
    d_dV = cuda.to_device(np.zeros((particle_dim, particle_dim, d), dtype='f4'))
    d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
    #d_smoothing = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))

    x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
    y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))

    lmbda = lmbda_2d if d == 2 else lmbda_3d

    print(d_pos.shape)
    print(x.shape)
    print(y.shape)

    #get_new_smoothing_lengths(d_pos[:,:,0,:], x, y, particle_mass, kernel_radius, tpb, bpg)

    #steps = int(tEnd/dt)
    steps = 500
    for i in range(1, steps):
        print(i)

        update_pos_vel_halfstep[bpg, tpb](d_pos, d_vel, d_dV, dt)
    
        get_new_smoothing_lengths(d_pos[:,:,0,:], x, y, particle_mass, kernel_radius, tpb, bpg, n_iter=20)
        smoothing_lengths = x[:,:,1]

        calc_density[bpg, tpb](d_pos, particle_mass, smoothing_lengths, d_rho)

        print('----------------------------------------')
        print(np.max(d_rho.copy_to_host()))
        print(np.min(d_rho.copy_to_host()))

        print(np.max(smoothing_lengths.copy_to_host()))
        print(np.min(smoothing_lengths.copy_to_host()))
    

        calc_dV_toystar[bpg, tpb](d_pos, d_vel, particle_mass, smoothing_lengths, eq_state_const, polytropic_idx, lmbda, viscosity, d_rho, d_dV)
        update_pos_vel_halfstep[bpg, tpb](d_pos, d_vel, d_dV, dt)

        for i1 in range(particle_dim):
            for j1 in range(particle_dim):
                print('...')
                print(d_pos.copy_to_host()[i1,j1,0])
                print(d_dV.copy_to_host()[i1,j1])

        if i == 2:
            exit()

    plot_vals = d_pos.copy_to_host()[:,:,0].flatten()[:2]

    plt.plot(plot_vals)
    plt.show()

