import math
from numba import cuda
from kernels import w_quintic_gpu, dwdq_quintic_gpu


@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32, float32[:,:], float32, float32, float32, float32, float32[:,:], float32[:,:,:])')
def calc_dv_toystar(pos, vel, particle_mass, smoothing_lengths, eq_state_const, polytropic_idx, lmbda, viscosity, density, dV):
    i, j = cuda.grid(2)

    position = pos[i,j]#,0]
    velocity = vel[i,j]
    dim = len(position)

    rho = density[i,j]

    pressure = eq_state_const * (rho**(1+(1/polytropic_idx)))

    h_i = smoothing_lengths[i,j]

    for d in range(dim):
        dV[i,j,d] = 0

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

            other_rho = density[i1, j1]
            other_pressure = eq_state_const * (other_rho**(1+(1/polytropic_idx)))

            pressure_acc =  (pressure/(rho*rho)) + (other_pressure/(other_rho*other_rho))

            for d in range(dim):
                dV[i,j,d] += particle_mass * pressure_acc * grad_w * (position[d] - pos[i1, j1, d])

    for d in range(dim):
        dV[i,j,d] = -dV[i,j,d] - lmbda * position[d] - viscosity * velocity[d]


@cuda.jit('void(float32[:,:,:], float32, float32[:,:], float32[:,:])')
def calc_density(pos, particle_mass, smoothing_lengths, density):

    i, j = cuda.grid(2)

    position = pos[i,j]#,0]
    dim = len(position)

    h_i = smoothing_lengths[i,j]

    rho = 0.0

    for i1 in range(pos.shape[0]):
        for j1 in range(pos.shape[1]):

            radius = 0.0
            for d in range(dim):
                #radius += (position[d] - pos[i1, j1, 0, d])**2
                radius += (position[d] - pos[i1, j1, d])**2

            radius = math.sqrt(radius)
            h_j = smoothing_lengths[i1,j1]
            h = 0.5*(h_i + h_j)
            w = w_quintic_gpu(radius, h, dim)
            rho += particle_mass * w
    
    density[i, j] = rho

@cuda.jit('void(float32[:,:,:,:], float32[:,:,:,:], float32[:,:,:], float32)')
def update_pos_vel_halfstep(pos, vel, dV, dt):

    i, j = cuda.grid(2)

    position = pos[i,j,0]
    dim = len(position)

    for d in range(dim):
        pos_dim = pos[i,j,0,d]
        vel_dim = vel[i,j,0,d]

        dV_dim = dV[i,j,d]

        pos[i,j,1,d] = pos_dim
        vel[i,j,1,d] = vel_dim

        pos[i, j, 0, d] = pos_dim + 0.5 * dt * vel_dim
        vel[i, j, 0, d] = vel_dim + 0.5 * dt * dV_dim
    '''
    x_pos = pos[i, j, 0, 0]
    y_pos = pos[i, j, 0, 1]
    z_pos = pos[i, j, 0, 2]

    vel_x = vel[i,j,0,0]
    vel_y = vel[i,j,0,1]
    vel_z = vel[i,j,0,2]

    dV_x = dV[i,j,0]
    dV_y = dV[i,j,1]
    dV_z = dV[i,j,2]

    # save values
    pos[i,j,1,0] = x_pos
    pos[i,j,1,1] = y_pos
    pos[i,j,1,2] = z_pos

    vel[i,j,1,0] = vel_x
    vel[i,j,1,1] = vel_y
    vel[i,j,1,2] = vel_z

    pos[i, j, 0, 0] = x_pos + 0.5 * dt * vel_x
    pos[i, j, 0, 1] = y_pos + 0.5 * dt * vel_y
    pos[i, j, 0, 2] = z_pos + 0.5 * dt * vel_z

    vel[i, j, 0, 0] = vel_x + 0.5 * dt * dV_x
    vel[i, j, 0, 1] = vel_y + 0.5 * dt * dV_y
    vel[i, j, 0, 2] = vel_z + 0.5 * dt * dV_z
    '''