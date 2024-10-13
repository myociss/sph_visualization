import math
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import newton
from numba import cuda
from astro_core import calc_params_shocktube, calc_pressure_from_energy, calc_density, calc_mean_nd, leapfrog_update_nd, leapfrog_update


# from https://physics.stackexchange.com/questions/423758/how-to-get-exact-solution-to-sod-shock-tube-test:
def f(P, pL, pR, cL, cR, gg):
    a = (gg-1)*(cR/cL)*(P-1) 
    b = np.sqrt( 2*gg*(2*gg + (gg+1)*(P-1) ) )
    return P - pL/pR*( 1 - a/b )**(2.*gg/(gg-1.))

# from https://physics.stackexchange.com/questions/423758/how-to-get-exact-solution-to-sod-shock-tube-test:
def SodShockAnalytic(rL, uL, pL, rR, uR, pR, xs, x0, T, gg):
    # rL, uL, pL, rR, uR, pR : Initial conditions of the Reimann problem 
    # xs: position array (e.g. xs = [0,dx,2*dx,...,(Nx-1)*dx])
    # x0: THIS IS AN INDEX! the array index where the interface sits.
    # T: the desired solution time
    # gg: adiabatic constant 1.4=7/5 for a 3D diatomic gas
    dx = xs[1]
    Nx = len(xs)
    v_analytic = np.zeros((3,Nx),dtype='float64')

    # compute speed of sound
    cL = np.sqrt(gg*pL/rL); 
    cR = np.sqrt(gg*pR/rR);
    # compute P
    P = newton(f, 0.5, args=(pL, pR, cL, cR, gg), tol=1e-12);

    # compute region positions right to left
    # region R
    c_shock = uR + cR*np.sqrt( (gg-1+P*(gg+1)) / (2*gg) )
    x_shock = x0 + int(np.floor(c_shock*T/dx))
    v_analytic[0,x_shock-1:] = rR
    v_analytic[1,x_shock-1:] = uR
    v_analytic[2,x_shock-1:] = pR
    
    # region 2
    alpha = (gg+1)/(gg-1)
    c_contact = uL + 2*cL/(gg-1)*( 1-(P*pR/pL)**((gg-1.)/2/gg) )
    x_contact = x0 + int(np.floor(c_contact*T/dx))
    v_analytic[0,x_contact:x_shock-1] = (1 + alpha*P)/(alpha+P)*rR
    v_analytic[1,x_contact:x_shock-1] = c_contact
    v_analytic[2,x_contact:x_shock-1] = P*pR
    
    # region 3
    r3 = rL*(P*pR/pL)**(1/gg);
    p3 = P*pR;
    c_fanright = c_contact - np.sqrt(gg*p3/r3)
    x_fanright = x0 + int(np.ceil(c_fanright*T/dx))
    v_analytic[0,x_fanright:x_contact] = r3;
    v_analytic[1,x_fanright:x_contact] = c_contact;
    v_analytic[2,x_fanright:x_contact] = P*pR;
    
    # region 4
    c_fanleft = -cL
    x_fanleft = x0 + int(np.ceil(c_fanleft*T/dx))
    u4 = 2 / (gg+1) * (cL + (xs[x_fanleft:x_fanright]-xs[x0])/T )
    v_analytic[0,x_fanleft:x_fanright] = rL*(1 - (gg-1)/2.*u4/cL)**(2/(gg-1));
    v_analytic[1,x_fanleft:x_fanright] = u4;
    v_analytic[2,x_fanleft:x_fanright] = pL*(1 - (gg-1)/2.*u4/cL)**(2*gg/(gg-1));

    # region L
    v_analytic[0,:x_fanleft] = rL
    v_analytic[1,:x_fanleft] = uL
    v_analytic[2,:x_fanleft] = pL

    return v_analytic

particle_dim = 32
spatial_dim = 1
n_particles = particle_dim*particle_dim
threads = 16
tpb = (threads, threads)
bpg = ( int(particle_dim / threads), int(particle_dim / threads) )

n_right = n_particles // 5
dx_left = 0.6 / (n_particles - n_right)
dx_right = 0.6 / n_right
left = np.array([i*dx_left - 0.6 + dx_left for i in range(n_particles - n_right)])
right = np.array([(i - (n_particles - n_right))*dx_right + 0.5*dx_right for i in range((n_particles-n_right), n_particles)])

h = 2*(right[1] - right[0])

x = np.append(left, right)

gamma = 1.4
particle_mass = 0.75 / n_particles

rho = np.append(np.ones_like(left), np.ones_like(right)*0.25)
p = np.append(np.ones_like(left), np.ones_like(right)*0.1795)
e = p / ((gamma - 1) * rho)
v = np.zeros_like(x)

n_boundary = 35
mask = np.zeros(x.shape) + 1.0
mask[:n_boundary] = 0.0
mask[-n_boundary:] = 0.0

d_pos = cuda.to_device(np.reshape(x, (particle_dim, particle_dim, spatial_dim)).astype('f4'))
pos_i = cuda.to_device(np.zeros(d_pos.shape, dtype='f4'))
d_vel = cuda.to_device(np.zeros(d_pos.shape, dtype='f4'))
vel_i = cuda.to_device(np.zeros(d_vel.shape, dtype='f4'))
vel_mean =cuda.to_device( np.zeros(d_vel.shape, dtype='f4'))


d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
d_e = cuda.to_device(np.reshape(e, (particle_dim, particle_dim)).astype('f4'))
e_i = cuda.to_device(np.zeros(d_e.shape, dtype='f4'))
d_pressure = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
d_de = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
d_dv = cuda.to_device(np.zeros((particle_dim, particle_dim, spatial_dim), dtype='f4'))
d_mask = cuda.to_device(np.reshape(mask, (particle_dim, particle_dim)).astype('f4'))

smoothing_length = np.zeros((particle_dim, particle_dim)) + h
smoothing_length = cuda.to_device(smoothing_length.astype('f4'))

dt = 1e-04
tEnd = 0.2
steps = int(tEnd/dt)

for i in range(steps):
    print(i)

    pos_i[:] = d_pos
    vel_i[:] = d_vel
    e_i[:] = d_e

    leapfrog_update_nd[bpg, tpb](d_pos, d_pos, d_vel, dt*0.5)
    leapfrog_update_nd[bpg, tpb](d_vel, d_vel, d_dv, dt*0.5)

    calc_density[bpg, tpb](d_pos, particle_mass, smoothing_length, d_rho)
    calc_pressure_from_energy[bpg, tpb](d_rho, d_e, gamma, d_pressure)
    calc_params_shocktube[bpg, tpb](d_pos, d_vel, particle_mass, smoothing_length, gamma, d_rho, d_pressure, d_dv, d_de, d_mask)

    leapfrog_update_nd[bpg, tpb](vel_i, d_vel, d_dv, dt)
    leapfrog_update[bpg, tpb](e_i, d_e, d_de, dt)

    calc_mean_nd[bpg, tpb](vel_i, d_vel, vel_mean)
    leapfrog_update_nd[bpg, tpb](pos_i, d_pos, vel_mean, dt)

pos = np.reshape(d_pos.copy_to_host(), (particle_dim*particle_dim))
vel = np.reshape(d_vel.copy_to_host(), (particle_dim*particle_dim))
rho = np.reshape(d_rho.copy_to_host(), (particle_dim*particle_dim))
pressure = np.reshape(d_pressure.copy_to_host(), (particle_dim*particle_dim))

indexes = np.where((x > -0.4) & (x < 0.4))
    
Nx = 100
X = 1.
dx = X/(Nx-1)
xs = np.linspace(0,0.8,Nx)
x0 = Nx//2
analytic = SodShockAnalytic(1.0, 0.0, 1.0, 0.25, 0.0, 0.1795, xs, x0, steps*1e-04, 1.4)

png_path = os.path.join(os.path.dirname(__file__), f'figures/03_shocktube')

for param, numeric, analytic in zip(['density', 'velocity', 'pressure'], [rho, vel, pressure], analytic):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(pos[indexes], numeric[indexes], marker='.')
    ax1.plot(xs - 0.4,analytic.T, color='r')
    ax1.set_title(param)
    plt.savefig(os.path.join(png_path, f'{param}.png'))
    plt.close(fig)
