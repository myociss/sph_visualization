import math
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import cuda
from PIL import Image

from astro_core import calc_dv_polytrope, calc_density, leapfrog_update, leapfrog_update_nd, calc_mean_nd, calc_dv_polytrope_save, calc_pressure_from_energy, calc_density_masked
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths
from pmocz_functions import plot_frame, get_img, get_img_alt
from kernels import grav_grad_quintic_gpu, dwdq_quintic_gpu

@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:], float32[:,:], float32[:,:])')
def calc_divergence_curl(pos, vel, particle_mass, smoothing_lengths, density, divergence, curl):
    i, j = cuda.grid(2)

    position = pos[i,j]
    velocity = vel[i,j]
    dim = len(position)

    rho = density[i,j]

    h_i = smoothing_lengths[i,j]

    divv = 0.0

    curl0 = 0.0
    curl1 = 0.0
    curl2 = 0.0

    for i1 in range(pos.shape[0]):
        for j1 in range(pos.shape[1]):
            radius = 0.0
            for d in range(dim):
                radius += (position[d] - pos[i1, j1, d])**2

            radius = math.sqrt(radius)
            
            h_j = smoothing_lengths[i1,j1]
            h = 0.5*(h_i + h_j)

            if radius > 1e-12:
                grad_w = dwdq_quintic_gpu(radius, h, dim) * (1./h) / radius
            else:
                grad_w = 0.0

            grad0 = (position[0] - pos[i1, j1, 0]) * grad_w
            grad1 = (position[1] - pos[i1, j1, 1]) * grad_w
            grad2 = (position[2] - pos[i1, j1, 2]) * grad_w

            dvel0 = velocity[0] - vel[i1,j1,0]
            dvel1 = velocity[1] - vel[i1,j1,1]
            dvel2 = velocity[2] - vel[i1,j1,2]

            div0 = dvel0 * grad0
            div1 = dvel1 * grad1
            div2 = dvel2 * grad2

            #for d in range(dim):
            #    divv += (particle_mass[i1,j1]/density[i1,j1]) * (position[d] - pos[i1, j1, d]) * (velocity[d] - vel[i1, j1, d]) * grad_w

            divv += (particle_mass[i1,j1]/density[i1,j1]) * (div0 + div1 + div2)

            #curlv0 += (particle_mass[i1,j1]/density[i1,j1]) * ((velocity[1] - vel[i1,j1,1]) * (velocity[2] - vel[i1, j1, 2]) * grad_w
            #            - (vi[2] - vj[2]) * dWdx[1])

            curl0 += (particle_mass[i1,j1]/density[i1,j1]) * (dvel1*grad2 - dvel2*grad1)
            curl1 += (particle_mass[i1,j1]/density[i1,j1]) * (dvel2*grad0 - dvel0*grad2)
            curl2 += (particle_mass[i1,j1]/density[i1,j1]) * (dvel0*grad1 - dvel1*grad0)

    divergence[i,j] = abs(divv)
    curl[i,j] = math.sqrt(curl0*curl0 + curl1*curl1 + curl2*curl2)

            




@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32, float32[:,:], float32[:,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:], float32[:,:])')
def calc_params(pos, vel, particle_mass, smoothing_lengths, adiabatic_idx, density, pressure, dV, dE, mask, divergence, curl):
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

    f_i = divergence[i,j] / (divergence[i,j] + curl[i,j] + 0.0001*(c/h_i))

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

            #f_j = divergence[i1,j1] / (divergence[i1,j1] + curl[i1,j1] + 0.0001*(other_c/h_j))

            if v_r_dot < 0:
                visc = (-alpha_visc*c_mean*mu + beta_visc*mu*mu) / rho_mean
            else:
                visc = 0.0

            #visc *= 0.5 * abs(f_i + f_j)

            #if math.isnan(visc):
            #    print('-----')
            #    print('here')
            #    if math.isnan(c_mean):
            #        print('here1')
            #    if math.isnan(mu):
            #        print('here2')
            #    if math.isnan(rho_mean):
            #        print('here3')

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



np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_3d_pos_80.npy')
with open(np_path,'rb') as f:
    pos_init = np.load(f)

np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_3d_pos_small_80.npy')
with open(np_path,'rb') as f:
    pos_init_small = np.load(f)


np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_3d_energy_80.npy')
with open(np_path,'rb') as f:
    energy_init = np.load(f)

np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_3d_energy_small_80.npy')
with open(np_path,'rb') as f:
    energy_init_small = np.load(f)


particle_dim = pos_init.shape[0]
spatial_dim = 3#pos_init.shape[1]


np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_3d_mass_80.npy')
with open(np_path,'rb') as f:
    mass = np.load(f)

print(pos_init.shape)
print(energy_init.shape)


N = particle_dim*particle_dim

print(N)

threads = 16
tpb = (threads, threads)
bpg = ( int(particle_dim / threads), int(particle_dim / threads) )

M = 1.898e30 #3d
R = 6.99e9

h_init = 0.1 * R#(R * 4/3)

particle_mass = cuda.to_device(mass.astype('f4'))#M/N

polytropic_idx = 1
gamma = 1+(1/polytropic_idx)

pos_init[int(3 * particle_dim / 4):] = pos_init_small[int(3 * particle_dim / 4):]
pos_init[int(3 * particle_dim / 4):] += np.array([R*5, R, 0.0])#np.array([R*5, R*3, 0.0])
pos_init[:int(3 * particle_dim / 4)] -= np.array([R*2.5, 0.0, 0.0])

energy_init[int(3 * particle_dim / 4):] = energy_init_small[int(3 * particle_dim / 4):]

'''
pos_init[int(particle_dim / 2):] = pos_init[:int(particle_dim / 2)] #+ np.array([R*4, 0.0, 0.0])
pos_init[int(particle_dim / 2):] += np.array([R*5, R*3, 0.0])
pos_init[:int(particle_dim / 2)] -= np.array([R*2.5, 0.0, 0.0])
energy_init[int(particle_dim / 2):] = energy_init[:int(particle_dim / 2)]
'''

#d_pos = cuda.to_device(np.reshape(pos_init, (particle_dim, particle_dim, spatial_dim)).astype('f4'))
d_pos = cuda.to_device(pos_init.astype('f4'))
pos_i = cuda.to_device(np.zeros(d_pos.shape, dtype='f4'))


#d_vel = cuda.to_device(np.reshape(vel_init, (particle_dim, particle_dim, spatial_dim)).astype('f4'))
#d_vel = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))
vel_init = np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4')
vel_init[int(3 * particle_dim / 4):,:,0] -= 9e5#5e6
d_vel = cuda.to_device(vel_init)

vel_i = cuda.to_device(np.zeros(d_vel.shape, dtype='f4'))
vel_mean =cuda.to_device( np.zeros(d_vel.shape, dtype='f4'))


d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
#d_e = cuda.to_device(np.reshape(energy_init, (particle_dim, particle_dim)).astype('f4'))
d_e = cuda.to_device(energy_init.astype('f4'))
e_i = cuda.to_device(np.zeros(d_e.shape, dtype='f4'))
d_pressure = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
d_de = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
#d_dv = cuda.to_device(np.reshape(dv_init, (particle_dim, particle_dim, spatial_dim)).astype('f4'))
d_dv = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))

d_div = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
d_curl = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))


#d_mask = cuda.to_device(np.ones((particle_dim, particle_dim)).astype('f4'))

mask_cpu = np.ones((particle_dim,particle_dim), dtype='f4')
#mask_cpu[int(particle_dim / 2):] = 0.0
cpu_indexes = np.argwhere(mask_cpu.reshape((particle_dim*particle_dim,)) > 0)
d_mask = cuda.to_device(mask_cpu)

x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))

smoothing_length = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4') + h_init)

r_vals = []
t_vals = []

dt = 0.05
t = 0.0
steps = 226

all_pos = np.zeros((N, spatial_dim, steps), dtype='f4')
all_rho = np.zeros((N, steps), dtype='f4')

imgs = []

for i in range(steps):
    print('--------------')
    print(i)

    pos_i[:] = d_pos
    vel_i[:] = d_vel
    e_i[:] = d_e

    leapfrog_update_nd[bpg, tpb](d_pos, d_pos, d_vel, dt*0.5)
    leapfrog_update_nd[bpg, tpb](d_vel, d_vel, d_dv, dt*0.5)

    get_new_smoothing_lengths(d_pos, x, y, particle_mass, 3.0, tpb, bpg, d_mask, n_iter=20)
    smoothing_length = x[:,:,1]

    smoothing_length_cpu = smoothing_length.copy_to_host().reshape((particle_dim*particle_dim,))
    smoothing_length_cpu = np.squeeze(smoothing_length_cpu[cpu_indexes])

    calc_density_masked[bpg, tpb](d_pos, d_mask, particle_mass, smoothing_length, d_rho)

    print(dt)
    #assert not np.isnan(d_e.copy_to_host()).any()
    #assert not np.isnan(d_pos.copy_to_host()).any()
    #assert not np.isnan(d_vel.copy_to_host()).any()


    #assert not np.isnan(d_rho.copy_to_host()).any()
    #print(np.max(d_rho.copy_to_host()))
    #exit()
    calc_pressure_from_energy[bpg, tpb](d_rho, d_e, gamma, d_pressure)
    #assert not np.isnan(d_pressure.copy_to_host()).any()

    #calc_divergence_curl[bpg, tpb](d_pos, d_vel, particle_mass, smoothing_length, d_rho, d_div, d_curl)
    #assert not np.isnan(d_div.copy_to_host()).any()
    #assert not np.isnan(d_curl.copy_to_host()).any()


    calc_params[bpg, tpb](d_pos, d_vel, particle_mass, smoothing_length, gamma, d_rho, d_pressure, d_dv, d_de, d_mask, d_div, d_curl)

    print(np.argwhere(np.isnan(d_dv.copy_to_host())))
    if np.isnan(d_dv.copy_to_host()).any():
        index = np.argwhere(np.isnan(d_dv.copy_to_host()))[0]

        print(d_pos.copy_to_host()[index[0], index[1]])
        print(d_vel.copy_to_host()[index[0], index[1]])
        print(d_rho.copy_to_host()[index[0], index[1]])
        print(d_pressure.copy_to_host()[index[0], index[1]])

        print( np.any(d_pressure.copy_to_host() < 0) )
        print( np.any(d_rho.copy_to_host() < 0) )

    #assert not np.isnan(d_dv.copy_to_host()).any()
    print(np.max(d_dv.copy_to_host()))
    print(np.min(d_dv.copy_to_host()))
    #assert not np.isnan(d_de.copy_to_host()).any()

    #assert not np.isnan(d_dv.copy_to_host()).any()

    step_dv = d_dv.copy_to_host().reshape((particle_dim*particle_dim,spatial_dim))
    step_dv = np.squeeze(step_dv[cpu_indexes])
    step_dv = np.linalg.norm(step_dv, axis=-1)

    leapfrog_update_nd[bpg, tpb](vel_i, d_vel, d_dv, dt)
    leapfrog_update[bpg, tpb](e_i, d_e, d_de, dt)

    calc_mean_nd[bpg, tpb](vel_i, d_vel, vel_mean)
    leapfrog_update_nd[bpg, tpb](pos_i, d_pos, vel_mean, dt)

    all_pos[:,:,i] = (d_pos.copy_to_host()).reshape((N, spatial_dim))
    all_rho[:,i] = (d_rho.copy_to_host()).reshape((N,))


    t += dt
    #step_dv = d_dv.copy_to_host()
    #norms = np.linalg.norm(step_dv, axis=-1)
    #dt = np.min( 0.3 * np.sqrt(smoothing_length.copy_to_host() / norms))

    dt = np.min( 0.1 * np.sqrt(smoothing_length_cpu / step_dv) )

    #dists = np.sqrt( np.square(all_pos[cpu_indexes,0,i]) + np.square(all_pos[cpu_indexes,1,i]) )
    #center = np.array([-5e7* t, 0.0, 0.0])
    #dists = np.linalg.norm(all_pos[cpu_indexes,:,i] - center, axis=-1)
    #r_max = np.max(dists)
    #print(r_max)
    #r_vals.append(r_max)
    #t_vals.append(t)

    #if i % 3 == 0:
    imgs.append(get_img_alt(all_pos[:,:,i], all_rho[:,i], R*4))

    if i==steps-1:
        np_path = os.path.join(os.path.dirname(__file__), f'data/collision_test_pos.npy')
        with open(np_path,'wb') as f:
            np.save(f, d_pos.copy_to_host())

    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #ax1.scatter( all_pos[cpu_indexes,0,i], all_pos[cpu_indexes,1,i], c=[1 for idx in range(len(all_rho[cpu_indexes,i]))], cmap=plt.cm.jet, s=10, alpha=0.5)
    #plt.show()

#plt.plot(t_vals, r_vals)
#plt.show()

gif_path = os.path.join(os.path.dirname(__file__), f'figures/05_collision/collision_{spatial_dim}d.gif')
imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=40, loop=0)