import math
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import cuda
from PIL import Image

from astro_core import calc_dv_polytrope, calc_density, calc_density_masked, leapfrog_update, leapfrog_update_nd, calc_mean_nd, calc_dv_polytrope_save, calc_pressure_from_energy
from kernels import w_quintic
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths
from pmocz_functions import plot_frame, get_img
from kernels import grav_grad_quintic_gpu, dwdq_quintic_gpu


@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32, float32[:,:], float32[:,:], float32[:,:,:], float32[:,:], float32[:,:])')
def calc_params(pos, vel, particle_mass, smoothing_lengths, adiabatic_idx, density, pressure, dV, dE, mask):
    i, j = cuda.grid(2)

    if mask[i,j] == 0:
        return

    alpha_visc = 1
    beta_visc = 2
    epsilon_visc = 0.01

    position = pos[i,j]
    velocity = vel[i,j]
    dim = len(position)

    rho = density[i,j]
    p = pressure[i,j]
    c = math.sqrt(adiabatic_idx * p / rho)

    h_i = smoothing_lengths[i,j]

    d_e = 0.0

    for d in range(dim):
        dV[i,j,d] = 0


    for i1 in range(pos.shape[0]):
        for j1 in range(pos.shape[1]):
            if mask[i1,j1] == 0:
                continue

            radius = 0.0
            for d in range(dim):
                radius += (position[d] - pos[i1, j1, d])**2

            radius = math.sqrt(radius)
            
            h_j = smoothing_lengths[i1,j1]
            h = 0.5*(h_i + h_j)

            if radius > 1e-12:
                grad_w = dwdq_quintic_gpu(radius, h, dim) * (1./h) / radius
                grav_grad = grav_grad_quintic_gpu(radius, h) * (1./(h*h)) / radius
            else:
                grad_w = 0.0
                grav_grad = 0.0

            other_rho = density[i1, j1]
            other_p = pressure[i1, j1]
            other_c = math.sqrt(adiabatic_idx * other_p / other_rho)
            c_mean = 0.5 * (c + other_c)

            v_r_dot = 0.0

            for d in range(dim):
                v_r_dot += (position[d] - pos[i1, j1, d]) * (velocity[d] - vel[i1, j1, d])

            mu = h * v_r_dot / (radius*radius + epsilon_visc*h*h)
            rho_mean = 0.5 * (rho + other_rho)

            if v_r_dot < 0:
                visc = (-alpha_visc*c_mean*mu + beta_visc*mu*mu) / rho_mean
            else:
                visc = 0.0

            pressure_acc =  (p/(rho*rho)) + (other_p/(other_rho*other_rho))

            vij = 0.0

            for d in range(dim):
                grav_comp = 6.674e-8*particle_mass[i1,j1]*grav_grad*(position[d] - pos[i1, j1, d])
                dV[i,j,d] += particle_mass[i1,j1] * (pressure_acc + visc) * grad_w * (position[d] - pos[i1, j1, d]) + grav_comp

                vij += (velocity[d] - vel[i1, j1, d]) * grad_w * (position[d] - pos[i1, j1, d])

            d_e += particle_mass[i1,j1] * (pressure_acc + visc) * vij

    for d in range(dim):
        dV[i,j,d] = -dV[i,j,d] * mask[i,j]
    dE[i,j] = 0.5 * d_e * mask[i,j]

np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_3d_pos.npy')
with open(np_path,'rb') as f:
    pos_init = np.load(f)

np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_3d_energy.npy')
with open(np_path,'rb') as f:
    energy_init = np.load(f)

np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_3d_mass.npy')
with open(np_path,'rb') as f:
    mass = np.load(f)

print(pos_init.shape)
print(energy_init.shape)
print(mass.shape)
print(energy_init)
#exit()

#print(energy_init)
R = 6.99e9
spatial_dim = 3
polytropic_idx = 1
gamma = 1+(1/polytropic_idx)
h_init = R/ 5.

'''
spatial_dim = 3
R = 6.99e9
star_mass = 1.998e33
h_init = R/ 5.
eq_state_const = 2.6e12
polytropic_idx = 1
gamma = 1+(1/polytropic_idx)

mass[0,0] = star_mass

pos_init[0,0] = 3.123784414852968e5
pos_init[0,1] = 1.702036924966430e4
pos_init[0,2] = 0.0

pos_init[1:,0] += -R #-9.486838571279353e10
pos_init[1:,1] += -R#-5.169034544361552e9

#star_dens = mass[-1,-1] * w_quintic(0.0, h_init, 3)
#star_pressure = eq_state_const * (star_dens**gamma)
#star_energy = star_pressure / ((gamma - 1) * star_dens)
energy_init[0] = 0.0#star_energy

print(energy_init)
'''

dt = 0.05

particle_dim = pos_init.shape[0]#int(np.sqrt(pos_init.shape[0]))
N = particle_dim*particle_dim

print(N)

pos_init[int(particle_dim / 2):,:] = pos_init[:int(particle_dim / 2),:]
pos_init[:int(particle_dim / 2), :, 0] -= R * 1000
pos_init[int(particle_dim / 2):, :, 0] += R * 1000

#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.scatter( pos_init[:,:,0], pos_init[:,:,1], c=[1 for idx in range(len(pos_init))], cmap=plt.cm.jet, s=10, alpha=0.5)
#plt.show()



energy_init[int(particle_dim / 2):] = energy_init[:int(particle_dim / 2)]

#print(pos_init)
#print(energy_init)
print(mass)
print(np.sum(mass))
print(np.sum(mass)/2)
print(energy_init)
assert not np.isnan(energy_init).any()
#exit()


mask_cpu = np.ones((particle_dim,particle_dim), dtype='f4')
mask_cpu[int(particle_dim / 2):] = 0.0
mask = cuda.to_device(mask_cpu)
cpu_indexes = np.argwhere(mask_cpu.reshape((particle_dim*particle_dim,)) > 0)
print(cpu_indexes)
#exit()

d_pos = cuda.to_device(pos_init.astype('f4'))
pos_i = cuda.to_device(np.zeros(d_pos.shape, dtype='f4'))
#d_pos = cuda.to_device(np.reshape(pos_init, (particle_dim, particle_dim, spatial_dim)).astype('f4'))
#pos_i = cuda.to_device(np.reshape(pos_init, (particle_dim, particle_dim, spatial_dim)).astype('f4'))#cuda.to_device(np.zeros(d_pos.shape, dtype='f4'))

#d_vel = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))
#vel = np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4')
#vel[0,0,0] = -1.205460224677914e2
#vel[0,0,1] = 1.198395092992375e2
#vel[1:,1:,0] = 3.660946159165656e7
#vel[1:,1:,1] = -3.639489568414117e7
vel = np.zeros(pos_init.shape)
#vel[int(particle_dim / 2):, 0] = -5e2
vel = np.reshape(vel, d_pos.shape)
d_vel = cuda.to_device(vel.astype('f4'))
print(d_vel.dtype)
#exit()

print(d_vel.copy_to_host()[0,:,:])
print(d_pos.copy_to_host()[0,:,:])

#exit()

vel_i = cuda.to_device(vel) #cuda.to_device(np.zeros(d_vel.shape, dtype='f4'))
vel_mean = cuda.to_device(vel)#cuda.to_device( np.zeros(d_vel.shape, dtype='f4'))


d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
#d_e = cuda.to_device(np.reshape(energy_init, (particle_dim, particle_dim)).astype('f4'))
d_e = cuda.to_device(energy_init.astype('f4'))
e_i = cuda.to_device(np.zeros(d_e.shape, dtype='f4'))
d_pressure = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
d_de = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
d_dv = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))

smoothing_length = np.zeros((particle_dim, particle_dim)) + h_init
smoothing_length = cuda.to_device(smoothing_length.astype('f4'))

d_mass = cuda.to_device(mass.astype('f4'))

print(mass)
print(np.sum(mass))
print(np.sum(mass)/2)
#exit()

steps = 501

all_pos = np.zeros((N, spatial_dim, steps), dtype='f4')
all_rho = np.zeros((N, steps), dtype='f4')

threads = 16
tpb = (threads, threads)
bpg = ( int(particle_dim / threads), int(particle_dim / threads) )


x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))

dt = 0.0001

for i in range(steps):
    print(i)

    pos_i[:] = d_pos
    vel_i[:] = d_vel
    e_i[:] = d_e

    leapfrog_update_nd[bpg, tpb](d_pos, d_pos, d_vel, dt*0.5)
    leapfrog_update_nd[bpg, tpb](d_vel, d_vel, d_dv, dt*0.5)

    get_new_smoothing_lengths(d_pos, x, y, d_mass, 3.0, tpb, bpg, mask, n_iter=15)
    smoothing_length = x[:,:,1]

    smoothing_length_cpu = smoothing_length.copy_to_host().reshape((particle_dim*particle_dim,))
    smoothing_length_cpu = np.squeeze(smoothing_length_cpu[cpu_indexes])

    calc_density_masked[bpg, tpb](d_pos, mask, d_mass, smoothing_length, d_rho)
    calc_pressure_from_energy[bpg, tpb](d_rho, d_e, gamma, d_pressure)
    calc_params[bpg, tpb](d_pos, d_vel, d_mass, smoothing_length, gamma, d_rho, d_pressure, d_dv, d_de, mask)

    assert not np.isnan(d_pos.copy_to_host()).any()
    assert not np.isnan(d_rho.copy_to_host()).any()
    assert not np.isnan(d_pressure.copy_to_host()).any()
    #print(d_dv.copy_to_host())
    print(np.min(d_dv.copy_to_host()))
    #print(np.max(d_pressure.copy_to_host()))
    #print(np.min(d_pressure.copy_to_host()))
    assert not np.isnan(d_dv.copy_to_host()).any()
    assert not np.isnan(d_de.copy_to_host()).any()

    leapfrog_update_nd[bpg, tpb](vel_i, d_vel, d_dv, dt)
    leapfrog_update[bpg, tpb](e_i, d_e, d_de, dt)

    calc_mean_nd[bpg, tpb](vel_i, d_vel, vel_mean)
    leapfrog_update_nd[bpg, tpb](pos_i, d_pos, vel_mean, dt)

    all_pos[:,:,i] = (d_pos.copy_to_host()).reshape((N, spatial_dim))
    all_rho[:,i] = (d_rho.copy_to_host()).reshape((N,))

    step_dv = d_dv.copy_to_host().reshape((particle_dim*particle_dim,spatial_dim))
    step_dv = np.squeeze(step_dv[cpu_indexes])
    step_dv = np.linalg.norm(step_dv, axis=-1)

    dt = np.min( 0.3 * np.sqrt(smoothing_length_cpu / step_dv) )

    if i%1 == 0:
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        #ax1.scatter( all_pos[:,0,i], all_pos[:,1,i], c=all_rho[:,i], cmap=plt.cm.jet, s=10, alpha=0.5)
        ax1.scatter( all_pos[cpu_indexes,0,i], all_pos[cpu_indexes,1,i], c=[1 for idx in range(len(all_rho[cpu_indexes,i]))], cmap=plt.cm.jet, s=10, alpha=0.5)
        #ax1.set_facecolor('black')
        #ax1.set_facecolor((.1,.1,.1))
        plt.show()