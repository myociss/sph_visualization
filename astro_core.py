import math
from numba import cuda
from kernels import w_quintic_gpu, dwdq_quintic_gpu


'''
@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32, float32[:,:], float32, float32[:,:], float32[:,:], float32[:,:,:], float32[:,:])')
def calc_dv_denergy(pos, vel, particle_mass, smoothing_lengths, adiabatic_idx, density, energy, dV, dE):

    i, j = cuda.grid(2)

    alpha_visc = 1
    beta_visc = 2
    epsilon_visc = 0.01

    position = pos[i,j]#,0]
    velocity = vel[i,j]
    dim = len(position)


    e = energy[i,j]
    dim = len(position)

    rho = density[i,j]
    pressure = (adiabatic_idx - 1) * e * rho
    c = math.sqrt(adiabatic_idx * pressure / rho)

    h_i = smoothing_lengths[i,j]

    for d in range(dim):
        dV[i,j,d] = 0
'''

@cuda.jit('void(float32[:,:], float32, float32, float32[:,:])')
def calc_pressure_polytrope(density, eq_state_const, polytropic_idx, pressure):
    i, j = cuda.grid(2)

    adiabatic_index = 1+(1/polytropic_idx)

    rho = density[i,j]
    p = eq_state_const * (rho**adiabatic_index)

    pressure[i,j] = p

@cuda.jit('void(float32[:,:], float32[:,:], float32, float32[:,:])')
def calc_pressure_from_energy(density, energy, adiabatic_idx, pressure):
    i, j = cuda.grid(2)

    rho = density[i,j]
    e = energy[i,j]

    p = (adiabatic_idx - 1) * e * rho
    #print(p)
    #print(p)
    #print(pressure[i,j])

    pressure[i,j] = p


@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32, float32[:,:], float32, float32[:,:], float32[:,:], float32[:,:,:], float32[:,:], float32[:,:])')
def calc_params_shocktube(pos, vel, particle_mass, smoothing_lengths, adiabatic_idx, density, pressure, dV, dE, mask):
    i, j = cuda.grid(2)

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
                dV[i,j,d] += particle_mass * (pressure_acc + visc) * grad_w * (position[d] - pos[i1, j1, d])
                vij += (velocity[d] - vel[i1, j1, d]) * grad_w * (position[d] - pos[i1, j1, d])

            d_e += particle_mass * (pressure_acc + visc) * vij

    for d in range(dim):
        dV[i,j,d] = -dV[i,j,d] * mask[i,j]#- viscosity * velocity[d]
    dE[i,j] = 0.5 * d_e * mask[i,j]



@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32, float32[:,:], float32, float32, float32, float32[:,:], float32[:,:,:])')
def calc_dv_polytrope(pos, vel, particle_mass, smoothing_lengths, eq_state_const, polytropic_idx, lmbda, density, dV):
    i, j = cuda.grid(2)

    alpha_visc = 1
    beta_visc = 2
    epsilon_visc = 0.01

    position = pos[i,j]#,0]
    velocity = vel[i,j]
    dim = len(position)

    adiabatic_index = 1+(1/polytropic_idx)

    rho = density[i,j]
    pressure = eq_state_const * (rho**adiabatic_index)
    #energy = pressure / ((adiabatic_index - 1) * rho)

    # eq_state_const * (rho**adiabatic_index) / ((adiabatic_index - 1) * rho)
    #c = math.sqrt(energy)
    c = math.sqrt(adiabatic_index * pressure / rho)

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
            other_pressure = eq_state_const * (other_rho**adiabatic_index)
            #other_energy = other_pressure / ((adiabatic_index - 1) * other_rho)
            #other_c = math.sqrt(other_energy)
            other_c = math.sqrt(adiabatic_index * other_pressure / other_rho)
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

            #print(visc)



            #d_r = 0.0
            #d_v = 0.0
            #for d in range(dim):
            #    d_r += 

            pressure_acc =  (pressure/(rho*rho)) + (other_pressure/(other_rho*other_rho))

            for d in range(dim):
                dV[i,j,d] += particle_mass * (pressure_acc + visc) * grad_w * (position[d] - pos[i1, j1, d])

    for d in range(dim):
        dV[i,j,d] = -dV[i,j,d] - lmbda * position[d] #- viscosity * velocity[d]


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


@cuda.jit('void(float32[:,:,:,:], float32[:,:,:,:], float32[:,:,:], float32[:,:,:], float32[:,:], float32)')
def update_quantities_halfstep(pos, vel, dV, energy, dE, dt):
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

    e = energy[i,j,0]
    energy[i,j,1] = e
    energy[i,j,0] = e + 0.5 * dt * dE[i,j]

