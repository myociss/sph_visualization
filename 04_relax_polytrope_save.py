import math
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import cuda
from scipy.special import gamma
from PIL import Image

from astro_core import calc_dv_polytrope, calc_density, leapfrog_update_nd, calc_mean_nd, calc_dv_polytrope_save
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths
from pmocz_functions import plot_frame, get_img


#R = 0.75 # star radius
R = 7000000 #/ 5#*2
solar_mass = 1.989e30
#M=0.1*solar_mass# for 2d#/ 5
M = 0.6*solar_mass # for 3d

#M = 0.6*solar_mass

eq_state_const = 0.1 * (R * 4/3)
#M = 0.2*solar_mass
#M = 2 # star mass

#R = 0.75
#M = 2
#eq_state_const = 0.1 * (R * 4/3)

# jupiter
'''
M = 0.001*solar_mass
R = 70000000
eq_state_const = 2.6e12
'''

polytropic_idx = 3/2


h_init = 0.1 * (R * 4/3)    # smoothing length
#eq_state_const = 0.005*0.1 * (R * 4/3) # equation of state constant
viscosity = 1 # damping

lmbda_2d = 2*eq_state_const*np.pi**(-1/polytropic_idx) * ( ( (M*(1+polytropic_idx)) / (R**2) )**(1 + 1/polytropic_idx) ) / M
lmbda_3d = 2*eq_state_const*(1+polytropic_idx)*np.pi**(-3/(2*polytropic_idx)) * (M*gamma(5/2+polytropic_idx)/R**3/gamma(1+polytropic_idx))**(1/polytropic_idx) / R**2

print(lmbda_2d)
print(lmbda_3d)

#exit()

tEnd = 8#12
#dt = 0.001#0.001#0.005#0.005


configs = [
    #(2, 32, lmbda_2d),
    #(2, 16, lmbda_2d),
    #(3, 32, lmbda_3d,4, 0.005),
    #(2, 48, lmbda_2d, 40)
    #(2, 32, lmbda_2d, 50, 0.0001)
    (3, 32, lmbda_2d, 50, 0.0001)
]

t = 0

import time



for spatial_dim, particle_dim, lmbda, downsample, dt in configs:

    dt_init = dt

    threads = 16
    tpb = (threads, threads)
    bpg = ( int(particle_dim / threads), int(particle_dim / threads) )

    mask = cuda.to_device(np.ones((particle_dim, particle_dim), dtype='f4'))


    N = particle_dim*particle_dim    # Number of particles

    smoothing_length = np.zeros((particle_dim, particle_dim)) + h_init
    smoothing_length = cuda.to_device(smoothing_length.astype('f4'))

    #init_pos = (R * 4/3) * np.random.randn(particle_dim, particle_dim, spatial_dim).astype('f4') # for 2d
    init_pos = (R * 4/3) *0.3* np.random.randn(particle_dim, particle_dim, spatial_dim).astype('f4') # for 3d
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
    for i in range(0, 501):
        #print(i)

        
        pos_i[:] = d_pos
        vel_i[:] = d_vel

        leapfrog_update_nd[bpg, tpb](d_pos, d_pos, d_vel, dt*0.5)
        leapfrog_update_nd[bpg, tpb](d_vel, d_vel, d_dV, dt*0.5)

        get_new_smoothing_lengths(d_pos, x, y, particle_mass, 3.0, tpb, bpg, mask, n_iter=15)
        smoothing_length = x[:,:,1]
        x_cpu = x.copy_to_host()
        delta_ratio = np.max(np.abs(x_cpu[:,:,2] - x_cpu[:,:,0]) / h_init)
        #assert np.all(delta_ratio < 0.02 * h_init)

        calc_density[bpg, tpb](d_pos, particle_mass, smoothing_length, d_rho)
        calc_dv_polytrope_save[bpg, tpb](d_pos, d_vel, particle_mass, smoothing_length, eq_state_const, polytropic_idx, lmbda, d_rho, d_dV)
        #calc_dv_polytrope_save[bpg, tpb](d_pos, d_vel, particle_mass, smoothing_length, eq_state_const, polytropic_idx, lmbda, d_rho, d_dV)

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
            imgs.append(get_img(all_pos[:,:,i], all_rho[:,i], R, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda)) #3d
            #imgs.append(get_img(all_pos[:,:,i], all_rho[:,i], R*3, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda)) #2d

        '''
        if i%400 == 0 or i == steps - 1:
        #if i==steps-1:

            cmap = plt.cm.jet#plt.cm.jet
            cval = all_rho[:,i]#np.minimum((all_rho[:,i]-3*(R*4/3))/3, 3 * R*4/3).flatten()

            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            ax1.scatter(all_pos[:,:,i][:,0],all_pos[:,:,i][:,1], c=cval, cmap=cmap, s=10, alpha=0.5)
            #ax1.scatter(pos[:,0],pos[:,1], c=cval, cmap=cmap, s=10, alpha=0.5)
            ax1.set(xlim=(-1.4 * (R * 4/3), 1.4 * (R * 4/3)), ylim=(-1.2 * (R * 4/3), 1.2 * (R * 4/3)))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-(R * 4/3),0,(R * 4/3)])
            ax1.set_yticks([-(R * 4/3),0,(R * 4/3)])
            ax1.set_facecolor('black')
            ax1.set_facecolor((.1,.1,.1))
            plt.show()
        '''


        '''
        #all_pos[:,:,i] = (d_pos.copy_to_host()).reshape((N, spatial_dim))
        #all_rho[:,i] = (d_rho.copy_to_host()).reshape((N,))

        pos = (d_pos.copy_to_host()).reshape((N, spatial_dim))
        dens = (d_rho.copy_to_host()).reshape((N,))

        #print(np.max(all_rho[:,i]))

        #imgs.append(get_img(all_pos[:,:,i], all_rho[:,i], R, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda))

        if i%1000 == 0 or i == steps - 1:
        #if i==steps-1:

            cmap = plt.cm.jet#plt.cm.jet
            cval = dens#all_rho[:,i]#np.minimum((all_rho[:,i]-3*(R*4/3))/3, 3 * R*4/3).flatten()

            fig = plt.figure()
            ax1 = fig.add_subplot(111)

            #ax1.scatter(all_pos[:,:,i][:,0],all_pos[:,:,i][:,1], c=cval, cmap=cmap, s=10, alpha=0.5)
            ax1.scatter(pos[:,0],pos[:,1], c=cval, cmap=cmap, s=10, alpha=0.5)
            ax1.set(xlim=(-1.4 * (R * 4/3), 1.4 * (R * 4/3)), ylim=(-1.2 * (R * 4/3), 1.2 * (R * 4/3)))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-(R * 4/3),0,(R * 4/3)])
            ax1.set_yticks([-(R * 4/3),0,(R * 4/3)])
            ax1.set_facecolor('black')
            ax1.set_facecolor((.1,.1,.1))
            plt.show()
        '''

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
    