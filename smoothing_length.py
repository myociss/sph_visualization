import math
import numpy as np
from numba import cuda
from kernels import w_quintic_gpu, dwdq_quintic_gpu, dwdh_quintic_gpu

eta_const = 1.0 # from "Phantom: A smoothed particle hydrodynamics and magnetohydrodynamics code for astrophysics"
# (https://arxiv.org/pdf/1702.03930) table 1


@cuda.jit('void(float32[:,:,:], float32[:,:], float32[:,:], float32[:,:], float32[:,:])')
def calc_zeta_prime(pos, mask, particle_mass, smoothing_lengths, zeta_prime):
    i, j = cuda.grid(2)

    if mask[i,j]==0:
        return

    position = pos[i, j]
    dim = len(position)
    h = smoothing_lengths[i,j]

    rho = 0.0

    gradh = 0.0

    for i1 in range(pos.shape[0]):
        for j1 in range(pos.shape[1]):
            if mask[i1,j1]==0:
                continue

            radius = 0.0
            for d in range(dim):
                radius += (position[d] - pos[i1, j1, d])**2

            radius = math.sqrt(radius)

            #dwdh = dwdh_quintic_gpu(radius, h, dim)
            #gradh += particle_mass[i1,j1] * dwdh

            w = w_quintic_gpu(radius, h, dim)

            if radius > 1e-12:
                grad_w = dwdq_quintic_gpu(radius, h, dim) * (1./h)
            else:
                grad_w = 0.0

            gradh +=  particle_mass[i1,j1] * ( (-radius/h)*grad_w - (3/h)*w)

            rho += particle_mass[i1,j1] * w

    # sort of close?????
    #omega = (1.0 - (3 * h / rho) * gradh  ) 
    #zeta_prime[i,j] = - 3 * rho * omega / h

    omega = (1.0 + (dim * h / rho) * gradh  )
    zeta_prime[i,j] = - (dim * rho / h) * omega
    #zeta_prime = gradh - (- rho / (3 * h))

    #omega = 1.0 + (h/(3*rho))*gradh
    #zeta_prime[i,j] =  3 * rho * omega / h

    #omega = (1.0 + (h/(3.0*rho)) )*gradh
    #zeta_prime[i,j] =  -3.0 * rho * omega / h
    
            

@cuda.jit('void(float32[:,:,:], float32[:,:], float32[:,:], float32[:,:], float32[:,:])')
def calc_zeta(pos, mask, particle_mass, smoothing_lengths, zeta):
    i, j = cuda.grid(2)

    if mask[i,j]==0:
        return

    position = pos[i, j]
    dim = len(position)
    h = smoothing_lengths[i,j]

    rho = 0.0

    for i1 in range(pos.shape[0]):
        for j1 in range(pos.shape[1]):
            if mask[i1,j1]==0:
                continue
            
            radius = 0.0
            for d in range(dim):
                radius += (position[d] - pos[i1, j1, d])**2

            radius = math.sqrt(radius)
            w = w_quintic_gpu(radius, h, dim)
            rho += particle_mass[i1,j1] * w

    #zeta[i,j] = particle_mass[i,j] * ( (eta_const / h)**dim ) - rho
    zeta[i,j] = rho - ( particle_mass[i,j] * ( (eta_const / h)**dim ) )


@cuda.jit('void(float32[:,:,:], float32[:,:], float32, float32[:,:,:])')
def calc_h_guesses(pos, mask, kernel_r, h_guesses):
    i, j = cuda.grid(2)

    if mask[i,j]==0:
        return

    position = pos[i, j]
    dim = len(position)

    min_radius = np.inf
    max_radius = 0.0

    for i1 in range(pos.shape[0]):
        for j1 in range(pos.shape[1]):
            if mask[i1,j1]==0:
                continue

            radius = 0.0
            for d in range(dim):
                radius += (position[d] - pos[i1, j1, d])**2

            radius = math.sqrt(radius)

            if radius > 0.0 and radius < min_radius:
                min_radius = radius
            if radius > max_radius:
                max_radius = radius

    h_guesses[i,j,0] = min_radius * (1 / kernel_r)
    h_guesses[i,j,2] = (max_radius + 1e-15) * (1 / kernel_r)

@cuda.jit('void(float32[:,:,:])')
def calc_midpoint(x):
    i, j = cuda.grid(2)
    x[i,j,1] = 0.5 * (x[i,j,0] + x[i,j,2])

@cuda.jit('void(float32[:,:,:], float32[:,:,:])')
def bisect_update(x, y):
    i, j = cuda.grid(2)
    
    same_sign = y[i,j,0] * y[i,j,1] > 0

    if same_sign:
        x[i,j,0] = x[i,j,1]
    else:
        x[i,j,2] = x[i,j,1]


def get_new_smoothing_lengths(pos, x, y, particle_mass, kernel_radius, tpb, bpg, mask, n_iter=30):
    #print(pos.shape)
    #print(x.shape)
    #print(y.shape)
    calc_h_guesses[bpg, tpb](pos, mask, kernel_radius, x)

    #print(pos.shape)
    #print(x.shape)
    #print(y.shape)
    
    calc_zeta[bpg, tpb](pos, mask, particle_mass, x[:,:,0], y[:,:,0])
    calc_zeta[bpg, tpb](pos, mask, particle_mass, x[:,:,2], y[:,:,2])

    for i in range(n_iter):
        calc_midpoint[bpg, tpb](x)
        calc_zeta[bpg, tpb](pos, mask, particle_mass, x[:,:,1], y[:,:,1])
        bisect_update[bpg, tpb](x, y)

    return 0