import math
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import cuda
from PIL import Image
from sklearn.neighbors import KDTree

from astro_core import calc_dv_polytrope, calc_density, leapfrog_update, leapfrog_update_nd, calc_mean_nd, calc_dv_polytrope_save, calc_pressure_from_energy, calc_density_masked, calc_alpha_deriv
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths#, get_new_smoothing_lengths_new
from pmocz_functions import plot_frame, get_img, get_img_alt
from kernels import grav_grad_quintic_gpu, dwdq_quintic_gpu, w_quintic


@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32, float32[:,:], float32[:,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:])')
def calc_params(pos, vel, particle_mass, smoothing_lengths, adiabatic_idx, density, pressure, dV, dE, alpha, mask):
    i, j = cuda.grid(2)

    if mask[i,j] == 0:
        return

    alpha_visc = 1.0
    beta_visc = 2*alpha_visc
    #alpha_visc_i = alpha[i,j]
    #alpha_visc = alpha_visc_i
    #beta_visc = 2 * alpha_visc
    epsilon_visc = 0.01

    position = pos[i,j]
    velocity = vel[i,j]
    dim = len(position)

    rho = density[i,j]
    p = pressure[i,j]
    c = math.sqrt(adiabatic_idx * p / rho)

    h_i = smoothing_lengths[i,j]

    d_e = 0.0

    #f_i = divergence[i,j] / (divergence[i,j] + curl[i,j] + 0.0001*(c/h_i))

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

            if math.isnan(visc):
                print('-----')
                print('here')
                if math.isnan(c_mean):
                    print('here1')
            #    if math.isnan(mu):
            #        print('here2')
            #    if math.isnan(rho_mean):
            #        print('here3')

            pressure_acc =  (p/(rho*rho)) + (other_p/(other_rho*other_rho))

            if pressure_acc + visc < 0:
                print('\n\n\n\n\n\n\n\n\nTHIS HAPPENS')
            if pressure_acc < 0:
                print('\n\n\n\n\n\n\n\n\nTHIS HAPPENS1')

            vij = 0.0

            for d in range(dim):
                grav_comp = 6.674e-8*particle_mass[i1,j1]*grav_grad*(position[d] - pos[i1, j1, d])
                dV[i,j,d] += particle_mass[i1,j1] * (pressure_acc + visc) * grad_w * (position[d] - pos[i1, j1, d]) + grav_comp

                vij += (velocity[d] - vel[i1, j1, d]) * grad_w * (position[d] - pos[i1, j1, d])

            d_e += particle_mass[i1,j1] * (pressure_acc + visc) * vij

    for d in range(dim):
        dV[i,j,d] = -dV[i,j,d] * mask[i,j]
    
    #if d_e < 0:
    #    print('\n\n\n\n\n\n\n\n\nTHIS HAPPENS2')
    
    dE[i,j] = 0.5 * d_e * mask[i,j]


np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_3d_pos_128_full.npy')
with open(np_path,'rb') as f:
    pos_init = np.load(f)



np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_3d_energy_128_full.npy')
with open(np_path,'rb') as f:
    energy_init = np.load(f)


particle_dim = pos_init.shape[0]
spatial_dim = 3#pos_init.shape[1]


np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_3d_mass_128_full.npy')
with open(np_path,'rb') as f:
    mass = np.load(f)

print(pos_init.shape)
print(energy_init.shape)

print(mass)
print(mass[0,0] * (128 * 128 - 1))

sun_mass = 1.998e33

mass[0,0] = sun_mass
pos_init[0,0,0] = 0.5 * -94868698091.23502
pos_init[0,0,1] = 5.169034544361552e9

vel_init = np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4')
vel_init[:,:,0] = -3.660946159165656e7
vel_init[:,:,1] = 3.639489568414117e7
vel_init[0,0,0] = 0.0
vel_init[0,0,1] = 0.0

sun_pressure = 1.2e17 / 5
sun_polytropic_idx = 3
sun_adiabatic_index = 1+(1/sun_polytropic_idx)
sun_density = sun_mass * w_quintic(0.0, 1.0, 3)
sun_energy = sun_pressure / ((sun_adiabatic_index - 1) * sun_density)
print(sun_density)
energy_init[0,0] = sun_energy
#exit()

print(vel_init)


N = particle_dim*particle_dim

print(N)

threads = 16
tpb = (threads, threads)
bpg = ( int(particle_dim / threads), int(particle_dim / threads) )

M = 1.898e30 #3d
R = 6.99e9

h_init = 0.05 * R

particle_mass = cuda.to_device(mass.astype('f4'))#M/N

polytropic_idx = 1
gamma = 1+(1/polytropic_idx)

d_pos = cuda.to_device(pos_init.astype('f4'))
pos_i = cuda.to_device(np.zeros(d_pos.shape, dtype='f4'))


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

alpha_init = 0.9

d_alpha = cuda.to_device(np.zeros((particle_dim,particle_dim), dtype='f4') + alpha_init)
alpha_i = cuda.to_device(np.zeros(d_alpha.shape, dtype='f4'))
d_dalpha = cuda.to_device(np.zeros(d_alpha.shape, dtype='f4'))


#d_mask = cuda.to_device(np.ones((particle_dim, particle_dim)).astype('f4'))

mask_cpu = np.ones((particle_dim,particle_dim), dtype='f4')
#mask_cpu[int(particle_dim / 2):] = 0.0
cpu_indexes = np.argwhere(mask_cpu.reshape((particle_dim*particle_dim,)) > 0)
d_mask = cuda.to_device(mask_cpu)

x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))

smoothing_length = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4') + h_init)
smoothing_length_init = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4') + h_init)

r_vals = []
t_vals = []

dt = 0.05
t = 0.0
steps = 460#720

all_pos = np.zeros((N, spatial_dim, steps), dtype='f4')
all_rho = np.zeros((N, steps), dtype='f4')

imgs = []

t_total = 0.0
import time

n_neighbors = 300

for i in range(steps):
    print('--------------')
    print(i)

    pos_i[:] = d_pos
    vel_i[:] = d_vel
    e_i[:] = d_e
    alpha_i[:] = d_alpha

    leapfrog_update_nd[bpg, tpb](d_pos, d_pos, d_vel, dt*0.5)
    leapfrog_update_nd[bpg, tpb](d_vel, d_vel, d_dv, dt*0.5)

    #start = time.time()
    #pos_cpu = np.reshape(d_pos.copy_to_host(), (particle_dim*particle_dim, spatial_dim))
    #get_new_smoothing_lengths_new(d_pos, x, y, particle_mass, 3.0, tpb, bpg, d_mask, n_iter=15)
    '''
    pos_2d_cpu = d_pos.copy_to_host()
    #print(pos_cpu.flags['C_CONTIGUOUS'])
    tree = KDTree(pos_cpu)
    distances, indices = tree.query(pos_cpu, 300)
    print(indices.dtype)
    #indices = np.reshape( np.unravel_index(indices, (particle_dim, particle_dim), order='F'), (particle_dim, particle_dim, n_neighbors, 2))
    indices = np.reshape(indices, (particle_dim, particle_dim, n_neighbors))
    print(indices.shape)
    distances = np.reshape(distances, (particle_dim, particle_dim, n_neighbors))


    print('***')
    print(pos_cpu[0])
    for j in range(10):
        print(indices[0,0,j])
        idx = np.unravel_index(indices[0,0,j], (particle_dim,particle_dim))
        print(idx)
        pt = pos_2d_cpu[idx[0],idx[1]]
        print(pt)
        print(distances[0,0,j])
    '''
        
    #print(indices.shape)
    #print(distances.shape)
    #print(time.time() - start)


    #start = time.time()
    #get_new_smoothing_lengths(d_pos, x, y, particle_mass, 3.0, tpb, bpg, d_mask, n_iter=15)
    #smoothing_length = x[:,:,1]
    #print(time.time() - start)

    #print(smoothing_length.copy_to_host())
    #print(smoothing_length.copy_to_host()[0,13])
    #print(smoothing_length.copy_to_host()[79,62])

    print('smoothing length ratio')
    print(np.max(smoothing_length.copy_to_host() / h_init))
    print(np.min(smoothing_length.copy_to_host() / h_init))
    print(smoothing_length.copy_to_host() / h_init)

    smoothing_length_cpu = smoothing_length.copy_to_host().reshape((particle_dim*particle_dim,))
    smoothing_length_cpu = np.squeeze(smoothing_length_cpu[cpu_indexes])

    calc_density_masked[bpg, tpb](d_pos, d_mask, particle_mass, smoothing_length, d_rho)

    print(dt)
    print(t)
    #assert not np.isnan(d_e.copy_to_host()).any()
    #assert not np.isnan(d_pos.copy_to_host()).any()
    #assert not np.isnan(d_vel.copy_to_host()).any()

    ratios_flattened = (smoothing_length.copy_to_host() / h_init).flatten()

    print(np.percentile(ratios_flattened, 95))

    if (d_e.copy_to_host() < 0).any():
        print(sorted(ratios_flattened.tolist(), reverse=True)[:100])
        indexes = np.argwhere(d_e.copy_to_host() < 0)
        print(len(indexes))
        index = indexes[0]
        print(index)

        print((smoothing_length.copy_to_host() / h_init)[index[0], index[1]])
        print(d_pos.copy_to_host()[index[0], index[1]])
        print(d_vel.copy_to_host()[index[0], index[1]])
        print(d_rho.copy_to_host()[index[0], index[1]])
        print(d_pressure.copy_to_host()[index[0], index[1]])

        print( np.any(d_pressure.copy_to_host() < 0) )
        print( np.any(d_rho.copy_to_host() < 0) )

    assert not (d_e.copy_to_host() < 0).any()


    #assert not np.isnan(d_rho.copy_to_host()).any()
    #print(np.max(d_rho.copy_to_host()))
    #exit()
    calc_pressure_from_energy[bpg, tpb](d_rho, d_e, gamma, d_pressure)
    #assert not np.isnan(d_pressure.copy_to_host()).any()

    #calc_divergence_curl[bpg, tpb](d_pos, d_vel, particle_mass, smoothing_length, d_rho, d_div, d_curl)
    #assert not np.isnan(d_div.copy_to_host()).any()
    #assert not np.isnan(d_curl.copy_to_host()).any()

    #calc_alpha_deriv[bpg, tpb](d_pos, d_vel, particle_mass, smoothing_length, d_rho, d_pressure, gamma, d_alpha, d_dalpha)


    calc_params[bpg, tpb](d_pos, d_vel, particle_mass, smoothing_length, gamma, d_rho, d_pressure, d_dv, d_de, d_alpha, d_mask)

    print(np.argwhere(np.isnan(d_dv.copy_to_host())))

    assert not (d_pressure.copy_to_host() < 0).any()
    assert not (d_rho.copy_to_host() < 0).any()
    assert not (gamma * d_pressure.copy_to_host() / d_rho.copy_to_host() < 0.0).any()

    if np.isnan(d_dv.copy_to_host()).any():
        index = np.argwhere(np.isnan(d_dv.copy_to_host()))[0]

        print((smoothing_length.copy_to_host() / h_init)[index[0], index[1]])
        print(d_pos.copy_to_host()[index[0], index[1]])
        print(d_vel.copy_to_host()[index[0], index[1]])
        print(d_rho.copy_to_host()[index[0], index[1]])
        print(d_pressure.copy_to_host()[index[0], index[1]])

        print( np.any(d_pressure.copy_to_host() < 0) )
        print( np.any(d_rho.copy_to_host() < 0) )

    assert not np.isnan(d_dv.copy_to_host()).any()

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
    leapfrog_update[bpg, tpb](alpha_i, d_alpha, d_dalpha, dt)

    calc_mean_nd[bpg, tpb](vel_i, d_vel, vel_mean)
    leapfrog_update_nd[bpg, tpb](pos_i, d_pos, vel_mean, dt)

    all_pos[:,:,i] = (d_pos.copy_to_host()).reshape((N, spatial_dim))
    all_rho[:,i] = (d_rho.copy_to_host()).reshape((N,))


    t += dt
    #step_dv = d_dv.copy_to_host()
    #norms = np.linalg.norm(step_dv, axis=-1)
    #dt = np.min( 0.3 * np.sqrt(smoothing_length.copy_to_host() / norms))

    #if i > 220:
    #    dt = np.min( 0.005 * np.sqrt(smoothing_length_cpu / step_dv) )
    #else:
    dt = np.min( 0.1 * np.sqrt(smoothing_length_cpu / step_dv) )

    #dists = np.sqrt( np.square(all_pos[cpu_indexes,0,i]) + np.square(all_pos[cpu_indexes,1,i]) )
    #center = np.array([-5e7* t, 0.0, 0.0])
    #dists = np.linalg.norm(all_pos[cpu_indexes,:,i] - center, axis=-1)
    #r_max = np.max(dists)
    #print(r_max)
    #r_vals.append(r_max)
    #t_vals.append(t)

    #if i % 3 == 0:
    imgs.append(get_img_alt(all_pos[:,:,i], all_rho[:,i], R*10))

    #if i==steps-1:
    #    np_path = os.path.join(os.path.dirname(__file__), f'data/collision_test_pos.npy')
    #    with open(np_path,'wb') as f:
    #        np.save(f, d_pos.copy_to_host())

    #fig = plt.figure()
    #ax1 = fig.add_subplot(111)
    #ax1.scatter( all_pos[cpu_indexes,0,i], all_pos[cpu_indexes,1,i], c=[1 for idx in range(len(all_rho[cpu_indexes,i]))], cmap=plt.cm.jet, s=10, alpha=0.5)
    #plt.show()

#plt.plot(t_vals, r_vals)
#plt.show()

gif_path = os.path.join(os.path.dirname(__file__), f'figures/06_black_hole/black_hole_{spatial_dim}d.gif')
imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=40, loop=0)