import math
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import cuda
from scipy.special import gamma
from PIL import Image

from astro_core import calc_dv_denergy, calc_density, update_quantities_halfstep, calc_dv_polytrope
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths

particle_dim = 32
spatial_dim = 1

threads = 16
tpb = (threads, threads)
bpg = ( int(particle_dim / threads), int(particle_dim / threads) )

particle_mass = 0.0006546020507812501
adiabatic_idx = 1.4

n_right = particle_dim*particle_dim // 9
left = np.linspace(-0.5,0,particle_dim*particle_dim - n_right)

left_density = np.zeros(left.shape) + 1.0
left_pressure = np.zeros(left.shape) + 1.0
left_energy = left_pressure / ((adiabatic_idx - 1) * left_density)

right = np.linspace(0,0.5,n_right)

right_density = np.zeros(right.shape) + 0.125
right_pressure = np.zeros(right.shape) + 0.1
right_energy = right_pressure / ((adiabatic_idx - 1) * right_density)

#x = np.reshape(np.append(left, right), (particle_dim, particle_dim,1))
x = np.append(left, right)
x = x.astype('f4')

x = np.reshape(x, (particle_dim, particle_dim,spatial_dim))

energy = np.append(left_energy, right_energy)
energy = energy.astype('f4')
energy = np.reshape(energy, (particle_dim, particle_dim,))

init_energy = np.zeros((particle_dim, particle_dim, 2), dtype='f4')
init_energy[:,:,0] = energy

#print(init_energy)
#exit()

print(init_energy.shape)
print(init_energy.dtype)
print(x.dtype)
#exit()

init_pos = np.zeros((particle_dim, particle_dim, 2, spatial_dim), dtype='f4')
init_pos[:,:,0,:] = x

d_pos = cuda.to_device(init_pos)

d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
d_vel = cuda.to_device(np.zeros((particle_dim, particle_dim, 2, spatial_dim), dtype='f4'))
d_dV = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))
d_energy = cuda.to_device(init_energy)#cuda.to_device(np.zeros((particle_dim, particle_dim, 2), dtype='f4'))


# this isn't working because the initial energy is not set
d_dE = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))

x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))

#print(x)

tEnd = 0.2
dt = 0.0001

steps = int(tEnd/dt)

#for i in range(1, steps):
for i in range(1, 2):
    print(i)

    update_quantities_halfstep[bpg, tpb](d_pos, d_vel, d_dV, d_energy, d_dE, dt)
    get_new_smoothing_lengths(d_pos[:,:,0,:], x, y, particle_mass, 3.0, tpb, bpg, n_iter=15)
    smoothing_length = x[:,:,1]

    calc_density[bpg, tpb](d_pos[:,:,0,:], particle_mass, smoothing_length, d_rho)

    #print(d_dV.shape)
    print(d_energy[:,:,0].copy_to_host())
    #exit()

    calc_dv_denergy[bpg, tpb](d_pos[:,:,0,:], d_vel[:,:,0,:], particle_mass, smoothing_length, adiabatic_idx, d_rho, d_energy[:,:,0], d_dV, d_dE)
    #calc_dv_polytrope[bpg, tpb](d_pos[:,:,0,:], d_vel[:,:,0,:], particle_mass, smoothing_length, 0.3, 3.0/2, 1.5, d_rho, d_dV)

    #dE = d_dE.copy_to_host()
    #print(dE)

    dV = d_dV.copy_to_host()
    #print(dV)
    #print(np.max(dV))
    #print(np.min(dV))
    print(d_energy[:,:,0].copy_to_host())
    #exit()

    update_quantities_halfstep[bpg, tpb](d_pos, d_vel, d_dV, d_energy, d_dE, dt)

    #all_pos[:,:,i] = (d_pos.copy_to_host()[:,:,0,:]).reshape((N, spatial_dim))
    #all_rho[:,i] = (d_rho.copy_to_host()).reshape((N,))

pos = d_pos[:,:,0,:].copy_to_host()
pos = np.reshape(pos, (particle_dim*particle_dim,))

vel = d_vel[:,:,0,:].copy_to_host()
vel = np.reshape(vel, (particle_dim*particle_dim,))

rho = d_rho.copy_to_host()
rho = np.reshape(rho, (particle_dim*particle_dim,))

plt.scatter(pos, rho)
plt.show()

plt.scatter(pos, vel)
plt.show()