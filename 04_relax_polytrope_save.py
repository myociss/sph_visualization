import math
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import cuda
from scipy.special import gamma
from PIL import Image

from astro_core import calc_dv_polytrope, calc_density, leapfrog_update_nd, calc_mean_nd, calc_dv_polytrope_save, calc_density_masked
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths
from pmocz_functions import plot_frame, get_img


R = 0.75 # star radius
R = 7000000 #/ 5#*2
solar_mass = 1.989e30
M=0.1*solar_mass#/ 5#0.6*solar_mass

#M = 0.6*solar_mass

eq_state_const = 0.1 * (R * 4/3)
#M = 0.2*solar_mass
#M = 2 # star mass

#R = 0.75
#M = 2
#eq_state_const = 0.1 * (R * 4/3)

# jupiter

#M = 0.001*solar_mass
M = 1.898e30 #3d
#M = 1.898e20
R = 6.99e9
#R = 6.99e12


eq_state_const = 2.6e12 #3d
#eq_state_const = 2.6e5


polytropic_idx = 1#3/2


h_init = 0.1 * (R * 4/3)    # smoothing length
#eq_state_const = 0.005*0.1 * (R * 4/3) # equation of state constant
viscosity = 1 # damping

lmbda_2d = 2*eq_state_const*np.pi**(-1/polytropic_idx) * ( ( (M*(1+polytropic_idx)) / (R**2) )**(1 + 1/polytropic_idx) ) / M
lmbda_3d = 2*eq_state_const*(1+polytropic_idx)*np.pi**(-3/(2*polytropic_idx)) * (M*gamma(5/2+polytropic_idx)/R**3/gamma(1+polytropic_idx))**(1/polytropic_idx) / R**2

print(lmbda_2d)
print(lmbda_3d)

#exit()

#exit()

tEnd = 8#12
#dt = 0.001#0.001#0.005#0.005


configs = [
    #(2, 32, lmbda_2d),
    #(2, 16, lmbda_2d),
    #(3, 32, lmbda_3d,4, 0.005),
    #(2, 48, lmbda_2d, 40)
    #(2, 32, lmbda_2d, 50, 0.0001),
    
    (3, 128, lmbda_2d, 50, 0.0001),
    #(2, 32, lmbda_2d, 50, 0.0001)
]

t = 0

import time



for spatial_dim, particle_dim, lmbda, downsample, dt in configs:

    dt_init = dt

    threads = 16
    tpb = (threads, threads)
    bpg = ( int(particle_dim / threads), int(particle_dim / threads) )


    N = particle_dim*particle_dim    # Number of particles

    smoothing_length = np.zeros((particle_dim, particle_dim)) + h_init
    smoothing_length = cuda.to_device(smoothing_length.astype('f4'))


    #mask = cuda.to_device(np.ones((particle_dim,particle_dim), dtype='f4'))
    mask_cpu = np.ones((particle_dim,particle_dim), dtype='f4')
    #mask_cpu[int(3 * particle_dim / 4):] = 0.0
    mask_cpu[0,0] = 0.0
    mask = cuda.to_device(mask_cpu)
    cpu_indexes = np.argwhere(mask_cpu.reshape((particle_dim*particle_dim,)) > 0)

    #init_pos = (R * 4/3)* np.random.randn(particle_dim, particle_dim, spatial_dim).astype('f4') # for 2d
    
    
    init_pos = (R * 4/3)*0.5 * np.random.randn(particle_dim, particle_dim, spatial_dim).astype('f4') # for 3d
    with open('data/points_128_start.npy','rb') as f:
        pos = np.load(f)
        #init_pos[:int(3 * particle_dim / 4)] = pos.astype('f4')
        init_p = pos.astype('f4')
    
    #init_pos = (R * 4/3)  np.random.randn(particle_dim, particle_dim, spatial_dim).astype('f4')
    d_pos = cuda.to_device(init_pos)

    pos_i = cuda.to_device(np.zeros(d_pos.shape, dtype='f4'))

    d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
    #d_vel = cuda.to_device(np.zeros((particle_dim, particle_dim, 2, spatial_dim), dtype='f4'))
    d_vel = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))
    vel_i = cuda.to_device(np.zeros(d_vel.shape, dtype='f4'))
    vel_mean =cuda.to_device( np.zeros(d_vel.shape, dtype='f4'))

    d_dV = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))

    particle_mass = M/np.sum(mask)

    print(particle_mass)
    #exit()

    #print(particle_mass * (64*64))
    #exit()
    #steps = int(tEnd/dt)

    steps = 401

    particle_masses = np.zeros((particle_dim, particle_dim)) + particle_mass
    particle_masses = cuda.to_device(particle_masses.astype('f4'))

    np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_{spatial_dim}d_mass_128_full.npy')
    with open(np_path,'wb') as f:
        np.save(f, particle_masses)

    #print(particle_mass)
    #continue

    all_pos = np.zeros((N, spatial_dim, steps), dtype='f4')
    #all_vel = np.zeros((N, spatial_dim, steps), dtype='f4')
    #all_dv = np.zeros((N, spatial_dim, steps), dtype='f4')
    all_rho = np.zeros((N, steps), dtype='f4')

    x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
    y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))

    imgs = []

    start = time.time()

    r_vals = []
    dv_vals = []
    t_vals = []

    for i in range(0, steps):
        print(i)
        
        pos_i[:] = d_pos
        vel_i[:] = d_vel

        leapfrog_update_nd[bpg, tpb](d_pos, d_pos, d_vel, dt*0.5)
        leapfrog_update_nd[bpg, tpb](d_vel, d_vel, d_dV, dt*0.5)

        get_new_smoothing_lengths(d_pos, x, y, particle_masses, 3.0, tpb, bpg, mask, n_iter=15)
        smoothing_length = x[:,:,1]

        smoothing_length_cpu = smoothing_length.copy_to_host().reshape((particle_dim*particle_dim,))
        smoothing_length_cpu = np.squeeze(smoothing_length_cpu[cpu_indexes])


        #calc_density[bpg, tpb](d_pos, particle_mass, smoothing_length, d_rho)
        #calc_dv_polytrope_save[bpg, tpb](d_pos, d_vel, particle_mass, smoothing_length, eq_state_const, polytropic_idx, lmbda, d_rho, d_dV)
        calc_density_masked[bpg, tpb](d_pos, mask, particle_masses, smoothing_length, d_rho)
        calc_dv_polytrope_save[bpg, tpb](d_pos, d_vel, particle_masses, smoothing_length, eq_state_const, polytropic_idx, lmbda, d_rho, d_dV, mask)

        #step_dv = d_dV.copy_to_host()
        step_dv = d_dV.copy_to_host().reshape((particle_dim*particle_dim,spatial_dim))
        step_dv = np.squeeze(step_dv[cpu_indexes])
        step_dv = np.linalg.norm(step_dv, axis=-1)

        #assert np.all(0.3 * np.sqrt(h_init / np.abs(step_dv)) > dt)

        #update_param(vel_i, d_vel, d_dV, dt, tpb, bpg)
        leapfrog_update_nd[bpg, tpb](vel_i, d_vel, d_dV, dt)

        #get_mean_velocity(vel_i, d_vel, vel_mean, tpb, bpg)
        calc_mean_nd[bpg, tpb](vel_i, d_vel, vel_mean)
        #update_param(pos_i, d_pos, vel_mean, dt, tpb, bpg)
        leapfrog_update_nd[bpg, tpb](pos_i, d_pos, vel_mean, dt)

        all_pos[:,:,i] = (d_pos.copy_to_host()).reshape((N, spatial_dim))
        #all_vel[:,:,i] = (d_vel.copy_to_host()).reshape((N, spatial_dim))
        #all_dv[:,:,i] = step_dv.reshape((N, spatial_dim))
        all_rho[:,i] = (d_rho.copy_to_host()).reshape((N,))

        t += dt

        #dt = np.min( 0.2 * np.sqrt(h_init / np.abs(step_dv)) )

        norms = np.linalg.norm(step_dv, axis=-1)


        dv_vals.append(np.mean(norms.flatten()))
        #print(norms.shape)
        #exit()

        dt = np.min( 0.3 * np.sqrt(smoothing_length_cpu / step_dv) )

        #r_max = np.max(np.linalg.norm(all_pos[:,:,i]))
        #print(r_max)
        #print(np.linalg.norm(all_pos[:,:,i]).shape)


        #dists = np.sqrt( np.square(all_pos[:,0,i]) + np.square(all_pos[:,1,i]) )
        dists = np.sqrt(np.linalg.norm(all_pos[cpu_indexes,:,i], axis=-1))
        #print(dists.shape)
        #exit()
        r_max = np.max(dists)
        #print(r_max)
        #print(dists.shape)
        #exit()
        r_vals.append(r_max)
        t_vals.append(t)

        if i==steps-1:
            np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_{spatial_dim}d_pos_128_full.npy')
            with open(np_path,'wb') as f:
                np.save(f, d_pos.copy_to_host())

            np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_{spatial_dim}d_energy_128_full.npy')
            with open(np_path,'wb') as f:
                adiabatic_index = 1+(1/polytropic_idx)
                pressure_final = eq_state_const * (d_rho.copy_to_host()**adiabatic_index)

                energy_final = pressure_final / ((adiabatic_index - 1) * d_rho.copy_to_host())
                np.save(f, energy_final)


        if i % 25==0 or i==steps-1:#downsample == 0:
            print('------')
            print(i)
            print(t)
            print(np.mean(norms.flatten()))
            # this is the slowdown
            print(np.max(all_rho[:,i]))
            imgs.append(get_img( np.squeeze(all_pos[cpu_indexes,:,i]), np.squeeze(all_rho[cpu_indexes,i]), R, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda)) #3d
            #imgs.append(get_img(all_pos[:,:,i], all_rho[:,i], R*3, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda)) #2d

            '''
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

    rho_final = all_rho[:,i]
    pos_final = all_pos[:,:,i]
    #vel_final = all_vel[:,:,i]
    #dv_final = all_dv[:,:,i]

    adiabatic_index = 1+(1/polytropic_idx)
    pressure_final = eq_state_const * (rho_final**adiabatic_index)

    energy_final = pressure_final / ((adiabatic_index - 1) * rho_final)

    '''
    np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_{spatial_dim}d_pos.npy')
    with open(np_path,'wb') as f:
        np.save(f, pos_final)

    np_path = os.path.join(os.path.dirname(__file__), f'data/jupiter_relaxed_{spatial_dim}d_energy.npy')
    with open(np_path,'wb') as f:
        np.save(f, energy_final)
    '''


    #plt.scatter(t_vals, r_vals)
    plt.plot(t_vals, r_vals)
    plt.show()

    plt.plot(t_vals, dv_vals)
    plt.show()
    
    
    gif_path = os.path.join(os.path.dirname(__file__), f'figures/04_relax_polytrope/toystar_{spatial_dim}d.gif')
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=160, loop=0)

    png_path = os.path.join(os.path.dirname(__file__), f'figures/04_relax_polytrope/toystar_{spatial_dim}d.png')
    #fig = plot_frame(all_pos[:,:,-1], all_rho[:,-1], R, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda, colormap='jet')

    cmap = plt.cm.jet#plt.cm.jet
    cval = all_rho[cpu_indexes,i] #np.minimum((all_rho[:,i]-3*(R*4/3))/3, 3 * R*4/3).flatten()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter( np.squeeze(all_pos[cpu_indexes,0,i]), np.squeeze(all_pos[cpu_indexes,1,i]), c=cval, cmap=cmap, s=10, alpha=0.5)
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
    