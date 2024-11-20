import math
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import cuda
from scipy.special import gamma
from PIL import Image

from astro_core import calc_dv_polytrope, calc_density_masked, leapfrog_update_nd, calc_mean_nd, calc_dv_polytrope_save
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths
from pmocz_functions import plot_frame, get_img

def split_from_ratio(ratio):
    return 1 / (1 + (1/ratio))

def get_masks(particle_dim, ratio):
    mask_large = np.zeros((particle_dim, particle_dim), dtype='f4')
    mask_small = np.copy(mask_large)
    idx = int(np.ceil(particle_dim * split_from_ratio(ratio)))
    mask_large[idx:] = 1.0
    mask_small[:idx] = 1.0

    return mask_large, mask_small

solar_mass = 1.989e30

mass_2d = 0.1*solar_mass
particle_dim_2d = 48
# ratio of small star to large star
ratio_2d = 1/3
gravity_2d = 7.5085875e-07
tEnd_2d = 4.0
mask_2d_large, mask_2d_small = get_masks(particle_dim_2d, ratio_2d)
particle_mass_2d = mass_2d / np.sum(mask_2d_large)

print(np.sum(mask_2d_large))
print(np.sum(mask_2d_small))


mass_3d = 0.6*solar_mass
particle_dim_3d = 64
ratio_3d = 1/3
gravity_3d = 3.33715e-11
tEnd_3d = 20.0#0.00145#5#100.0
mask_3d_large, mask_3d_small = get_masks(particle_dim_3d, ratio_3d)
particle_mass_3d = mass_3d / np.sum(mask_3d_large)

print(np.sum(mask_3d_large))
print(np.sum(mask_3d_small))

R = 7000000
base_k = 0.1 * (R * 4/3) #/ 3
polytropic_idx = 3/2

init_r_2d = (R * 4/3)
init_r_3d = (R * 4/3) * 0.3



configs = [
    #(2, particle_mass_2d, gravity_2d, 1.0, mask_2d_large, init_r_2d, base_k, 'large'),
    (2, particle_mass_2d, gravity_2d, tEnd_2d, mask_2d_small, init_r_2d * ratio_2d, base_k * ratio_2d, 'small'),
    #(3, particle_mass_3d, gravity_3d, 40.0, mask_3d_large, init_r_3d, base_k, 'large'),
    #(3, particle_mass_3d, gravity_3d, 40.0, mask_3d_small, init_r_3d*ratio_3d, base_k * ratio_3d, 'small'),
]


vals_3d = []

for spatial_dim, particle_mass, gravity, tEnd, mask_cpu, init_r, eq_state_const, size in configs:
    print(particle_mass)
    #exit()

    particle_dim = mask_cpu.shape[0]
    threads = 16
    tpb = (threads, threads)
    bpg = ( int(particle_dim / threads), int(particle_dim / threads) )

    smoothing_length = np.zeros((particle_dim, particle_dim))
    smoothing_length = cuda.to_device(smoothing_length.astype('f4'))
    mask = cuda.to_device(mask_cpu)

    init_pos = init_r * np.random.randn(particle_dim, particle_dim, spatial_dim).astype('f4')
    d_pos = cuda.to_device(init_pos)

    pos_i = cuda.to_device(np.zeros(d_pos.shape, dtype='f4'))

    d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))

    d_vel = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))
    vel_i = cuda.to_device(np.zeros(d_vel.shape, dtype='f4'))
    vel_mean =cuda.to_device( np.zeros(d_vel.shape, dtype='f4'))

    d_dV = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))

    x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
    y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))

    t = 0.0

    all_pos = []
    all_rho = []
    timesteps = []

    dt = 0.0001

    cpu_indexes = np.argwhere(mask_cpu.reshape((particle_dim*particle_dim,)) > 0)

    t_vals = []
    r_vals = []

    while t < tEnd:
        timesteps.append(dt)

        pos_i[:] = d_pos
        vel_i[:] = d_vel

        leapfrog_update_nd[bpg, tpb](d_pos, d_pos, d_vel, dt*0.5)
        leapfrog_update_nd[bpg, tpb](d_vel, d_vel, d_dV, dt*0.5)

        get_new_smoothing_lengths(d_pos, x, y, particle_mass, 3.0, tpb, bpg, mask, n_iter=15)
        smoothing_length = x[:,:,1]
        #x_cpu = x.copy_to_host()
        #delta_ratio = np.max(np.abs(x_cpu[:,:,2] - x_cpu[:,:,0]) / h_init)

        smoothing_length_cpu = smoothing_length.copy_to_host().reshape((particle_dim*particle_dim,))
        smoothing_length_cpu = np.squeeze(smoothing_length_cpu[cpu_indexes])

        calc_density_masked[bpg, tpb](d_pos, mask, particle_mass, smoothing_length, d_rho)
        calc_dv_polytrope[bpg, tpb](d_pos, d_vel, gravity, particle_mass, smoothing_length, eq_state_const, polytropic_idx, d_rho, d_dV, mask)


        #assert np.all(0.3 * np.sqrt(h_init / np.abs(step_dv)) > dt)

        #update_param(vel_i, d_vel, d_dV, dt, tpb, bpg)
        leapfrog_update_nd[bpg, tpb](vel_i, d_vel, d_dV, dt)

        #get_mean_velocity(vel_i, d_vel, vel_mean, tpb, bpg)
        calc_mean_nd[bpg, tpb](vel_i, d_vel, vel_mean)
        #update_param(pos_i, d_pos, vel_mean, dt, tpb, bpg)
        leapfrog_update_nd[bpg, tpb](pos_i, d_pos, vel_mean, dt)

        #all_pos.append((d_pos.copy_to_host()).reshape((N, spatial_dim)))
        #all_rho.append((d_rho.copy_to_host()).reshape((N,)))

        pos_cpu = d_pos.copy_to_host().reshape((particle_dim*particle_dim, spatial_dim))#[np.argwhere(mask_cpu > 0)]
        #print(pos_cpu.shape)
        pos_cpu = np.squeeze(pos_cpu[np.argwhere(mask_cpu.reshape((particle_dim*particle_dim,)) > 0)])
        #print(pos_cpu.shape)

        all_pos.append(pos_cpu)

        rho_cpu = d_rho.copy_to_host().reshape((particle_dim*particle_dim,))
        rho_cpu = np.squeeze(rho_cpu[np.argwhere(mask_cpu.reshape((particle_dim*particle_dim,)) > 0)])
        #print(rho_cpu.shape)
        all_rho.append(rho_cpu)
        #exit()

        t += dt

        step_dv = d_dV.copy_to_host().reshape((particle_dim*particle_dim,spatial_dim))
        step_dv = np.squeeze(step_dv[cpu_indexes])
        step_dv = np.linalg.norm(step_dv, axis=-1)
        
        
        #print(step_dv.shape)
        #exit()
        
        dists = np.linalg.norm(pos_cpu, axis=-1)
        r_max = np.max(dists)

        t_vals.append(t)
        r_vals.append(r_max)

        dt = np.min( 0.3 * np.sqrt(smoothing_length_cpu / step_dv) )

        #print(dists.shape)
        #exit()

        

        print(t)

    plt.plot(t_vals, r_vals)
    plt.show()

    vals_3d.append((t_vals, r_vals))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    cval = all_rho[-1]

    ax1.scatter(all_pos[-1][:,0],all_pos[-1][:,1], c=cval, cmap=plt.cm.jet, s=10, alpha=0.5)
    #ax1.scatter(pos[:,0],pos[:,1], c=cval, cmap=cmap, s=10, alpha=0.5)
    #ax1.set(xlim=(-1.4 * (R * 4/3), 1.4 * (R * 4/3)), ylim=(-1.2 * (R * 4/3), 1.2 * (R * 4/3)))
    ax1.set(xlim=(-1.4 * (R * 4/3), 1.4 * (R * 4/3)), ylim=(-1.2 * (R * 4/3), 1.2 * (R * 4/3)))
    ax1.set_aspect('equal', 'box')
    ax1.set_xticks([-(R * 4/3),0,(R * 4/3)])
    ax1.set_yticks([-(R * 4/3),0,(R * 4/3)])
    ax1.set_facecolor('black')
    ax1.set_facecolor((.1,.1,.1))    

    plt.show()


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(vals_3d[0][0], vals_3d[0][1])
ax1.plot(vals_3d[1][0], vals_3d[1][1])
plt.show()


exit()

'''
t = 0



for spatial_dim, particle_dim, lmbda, downsample, dt in configs:

    dt_init = dt

    threads = 16
    tpb = (threads, threads)
    bpg = ( int(particle_dim / threads), int(particle_dim / threads) )


    N = particle_dim*particle_dim    # Number of particles

    smoothing_length = np.zeros((particle_dim, particle_dim))
    smoothing_length = cuda.to_device(smoothing_length.astype('f4'))

    mask_cpu = np.ones((particle_dim, particle_dim), dtype='f4')
    mask_cpu[particle_dim//2:,particle_dim//2:] = 0.0

    mask = cuda.to_device(mask_cpu)

    init_pos = (R * 4/3) * np.random.randn(particle_dim, particle_dim, spatial_dim).astype('f4') # for 2d
    #init_pos = (R * 4/3) *0.3* np.random.randn(particle_dim, particle_dim, spatial_dim).astype('f4') # for 3d
    #init_pos = (R * 4/3)  np.random.randn(particle_dim, particle_dim, spatial_dim).astype('f4')
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

    start = time.time()

    r_vals = []
    t_vals = []


    #for i in range(1, steps):
    for i in range(0, 1501):
        #print(i)

        
        pos_i[:] = d_pos
        vel_i[:] = d_vel

        leapfrog_update_nd[bpg, tpb](d_pos, d_pos, d_vel, dt*0.5)
        leapfrog_update_nd[bpg, tpb](d_vel, d_vel, d_dV, dt*0.5)

        get_new_smoothing_lengths(d_pos, x, y, particle_mass, 3.0, tpb, bpg, n_iter=15)
        smoothing_length = x[:,:,1]
        x_cpu = x.copy_to_host()
        delta_ratio = np.max(np.abs(x_cpu[:,:,2] - x_cpu[:,:,0]) / h_init)
        #assert np.all(delta_ratio < 0.02 * h_init)

        calc_density[bpg, tpb](d_pos, particle_mass, smoothing_length, d_rho)
        calc_dv_polytrope_save[bpg, tpb](d_pos, d_vel, particle_mass, smoothing_length, eq_state_const, polytropic_idx, lmbda, d_rho, d_dV)

        step_dv = d_dV.copy_to_host()

        #assert np.all(0.3 * np.sqrt(h_init / np.abs(step_dv)) > dt)

        #update_param(vel_i, d_vel, d_dV, dt, tpb, bpg)
        leapfrog_update_nd[bpg, tpb](vel_i, d_vel, d_dV, dt)

        #get_mean_velocity(vel_i, d_vel, vel_mean, tpb, bpg)
        calc_mean_nd[bpg, tpb](vel_i, d_vel, vel_mean)
        #update_param(pos_i, d_pos, vel_mean, dt, tpb, bpg)
        leapfrog_update_nd[bpg, tpb](pos_i, d_pos, vel_mean, dt)

        all_pos[:,:,i] = (d_pos.copy_to_host()).reshape((N, spatial_dim))
        all_rho[:,i] = (d_rho.copy_to_host()).reshape((N,))

        t += dt

        dt = np.min( 0.2 * np.sqrt(h_init / np.abs(step_dv)) )

        #r_max = np.max(np.linalg.norm(all_pos[:,:,i]))
        #print(r_max)
        #print(np.linalg.norm(all_pos[:,:,i]).shape)
        dists = np.sqrt( np.square(all_pos[:,0,i]) + np.square(all_pos[:,1,i]) )
        r_max = np.max(dists)
        #print(r_max)
        #print(dists.shape)
        #exit()
        r_vals.append(r_max)
        t_vals.append(t)


        if i % 25==0 or i==steps-1:#downsample == 0:
            print(i)
            print(t)
            print(i*dt_init)
            # this is the slowdown
            print(np.max(all_rho[:,i]))
            #imgs.append(get_img(all_pos[:,:,i], all_rho[:,i], R, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda)) #3d
            imgs.append(get_img(all_pos[:,:,i], all_rho[:,i], R*3, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda)) #2d


    #plt.scatter(t_vals, r_vals)
    plt.plot(t_vals, r_vals)
    plt.show()
    
    
    gif_path = os.path.join(os.path.dirname(__file__), f'figures/04_relax_polytrope/toystar_{spatial_dim}d.gif')
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=160, loop=0)

    png_path = os.path.join(os.path.dirname(__file__), f'figures/04_relax_polytrope/toystar_{spatial_dim}d.png')
    #fig = plot_frame(all_pos[:,:,-1], all_rho[:,-1], R, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda, colormap='jet')

    cmap = plt.cm.jet#plt.cm.jet
    cval = all_rho[:,i]#np.minimum((all_rho[:,i]-3*(R*4/3))/3, 3 * R*4/3).flatten()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(all_pos[:,:,i][:,0],all_pos[:,:,i][:,1], c=cval, cmap=cmap, s=10, alpha=0.5)
    #ax1.scatter(pos[:,0],pos[:,1], c=cval, cmap=cmap, s=10, alpha=0.5)
    #ax1.set(xlim=(-1.4 * (R * 4/3), 1.4 * (R * 4/3)), ylim=(-1.2 * (R * 4/3), 1.2 * (R * 4/3)))
    ax1.set(xlim=(-1.4 * (R * 4/3), 1.4 * (R * 4/3)), ylim=(-1.2 * (R * 4/3), 1.2 * (R * 4/3)))
    ax1.set_aspect('equal', 'box')
    ax1.set_xticks([-(R * 4/3),0,(R * 4/3)])
    ax1.set_yticks([-(R * 4/3),0,(R * 4/3)])
    ax1.set_facecolor('black')
    ax1.set_facecolor((.1,.1,.1))
    #plt.show()


    plt.savefig(png_path)
    plt.show()
    plt.close(fig)
'''