import math
from scipy.constants import gravitational_constant
from numba import cuda
from kernels import w_quintic_gpu, dwdq_quintic_gpu, grav_grad_quintic_gpu

G = gravitational_constant#*1.3

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
        dV[i,j,d] = -dV[i,j,d] * mask[i,j]
    dE[i,j] = 0.5 * d_e * mask[i,j]




@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32, float32, float32[:,:], float32, float32, float32[:,:], float32[:,:,:], float32[:,:])')
def calc_dv_polytrope(pos, vel, gravity, particle_mass, smoothing_lengths, eq_state_const, polytropic_idx, density, dV, mask):
    i, j = cuda.grid(2)

    if mask[i,j] == 0:
        return

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
            if mask[i1,j1] == 0:
                continue

            radius = 0.0
            for d in range(dim):
                radius += (position[d] - pos[i1, j1, d])**2

            radius = math.sqrt(radius)

            #if radius > 3.0:
            #    continue
            

            h_j = smoothing_lengths[i1,j1]
            h = 0.5*(h_i + h_j)

            if radius > 1e-12:
                grad_w = dwdq_quintic_gpu(radius, h, dim) * (1./h) / radius
                grav_grad = grav_grad_quintic_gpu(radius, h) * (1./(h*h)) / radius
            else:
                grad_w = 0.0
                grav_grad = 0.0

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

            #grav_grad = grav_grad_quintic_gpu(radius, h) * (1./(h*h)) / radius



            #d_r = 0.0
            #d_v = 0.0
            #for d in range(dim):
            #    d_r += 

            pressure_acc =  (pressure/(rho*rho)) + (other_pressure/(other_rho*other_rho))

            for d in range(dim):
                #grav_comp = 0.5*G*particle_mass*grav_grad * (position[d] - pos[i1, j1, d]) *25000 * 0.9 #* 5
                #grav_comp = 0.5*G*particle_mass*grav_grad * (position[d] - pos[i1, j1, d])
                grav_comp = gravity*particle_mass*grav_grad*(position[d] - pos[i1, j1, d])
                dV[i,j,d] += particle_mass * (pressure_acc + visc) * grad_w * (position[d] - pos[i1, j1, d]) + grav_comp

    for d in range(dim):
        dV[i,j,d] = -dV[i,j,d]# - lmbda * position[d]


@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32, float32[:,:], float32, float32, float32, float32[:,:], float32[:,:,:])')
def calc_dv_polytrope_save(pos, vel, particle_mass, smoothing_lengths, eq_state_const, polytropic_idx, lmbda, density, dV):
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

            #if radius > 3.0:
            #    continue
            

            h_j = smoothing_lengths[i1,j1]
            h = 0.5*(h_i + h_j)

            if radius > 1e-12:
                grad_w = dwdq_quintic_gpu(radius, h, dim) * (1./h) / radius
                grav_grad = grav_grad_quintic_gpu(radius, h) * (1./(h*h)) / radius
            else:
                grad_w = 0.0
                grav_grad = 0.0

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

            #grav_grad = grav_grad_quintic_gpu(radius, h) * (1./(h*h)) / radius



            #d_r = 0.0
            #d_v = 0.0
            #for d in range(dim):
            #    d_r += 

            pressure_acc =  (pressure/(rho*rho)) + (other_pressure/(other_rho*other_rho))

            for d in range(dim):
                #grav_comp = 0.5*G*particle_mass*grav_grad * (position[d] - pos[i1, j1, d]) *25000 * 0.9 #* 5
                grav_comp = 0.5*G*particle_mass*grav_grad * (position[d] - pos[i1, j1, d])
                dV[i,j,d] += particle_mass * (pressure_acc + visc) * grad_w * (position[d] - pos[i1, j1, d]) + grav_comp

    for d in range(dim):
        dV[i,j,d] = -dV[i,j,d]# - lmbda * position[d]



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

@cuda.jit('void(float32[:,:,:], float32[:,:], float32, float32[:,:], float32[:,:])')
def calc_density_masked(pos, mask, particle_mass, smoothing_lengths, density):

    i, j = cuda.grid(2)
    if mask[i,j] == 0:
        return

    position = pos[i,j]#,0]
    dim = len(position)

    h_i = smoothing_lengths[i,j]

    rho = 0.0

    for i1 in range(pos.shape[0]):
        for j1 in range(pos.shape[1]):
            if mask[i1,j1] == 0:
                continue

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

@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:], float32)')
def leapfrog_update(x_current, x_next, dx, dt):
    # update 1 dimensional quantity (energy)
    i, j = cuda.grid(2)
    x_next[i,j] = x_current[i,j] + dt * dx[i,j]

@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32[:,:,:], float32)')
def leapfrog_update_nd(x_current, x_next, dx, dt):
    # update 2 or 3 dimensional quantity (position, velocity)
    i, j = cuda.grid(2)
    dim = len(x_current[i,j])
    for d in range(dim):
        x_next[i,j,d] = x_current[i,j,d] + dt * dx[i,j,d]

@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32[:,:,:])')
def calc_mean_nd(x0, x1, y):
    i, j = cuda.grid(2)
    dim = len(x0[i,j])
    for d in range(dim):
        y[i,j,d] = 0.5 * (x0[i,j,d] + x1[i,j,d])


def copy_devarray(arr):
    new_arr = cuda.device_array_like(arr)
    new_arr[:] = arr
    return new_arr

def get_mean_velocity(x0, x1, y, tpb, bpg):
    assert len(x0.shape) == 3# and x1.shape == 3
    dim = x0.shape[2]
    for d in range(dim):
        calc_mean[bpg, tpb](x0[:,:,d], x1[:,:,d], y[:,:,d])

def update_param(x_current, x_next, dx, dt, tpb, bpg):
    if len(x_current.shape) == 2:
        # 1D quantity
        leapfrog_update[bpg, tpb](x_current, x_next, dx, dt)
    else:
        # nD quantity
        assert len(x_current.shape) == 3# and dx.shape == 3
        dim = x_current.shape[2]
        for d in range(dim):
            leapfrog_update[bpg, tpb](x_current[:,:,d], x_next[:,:,d], dx[:,:,d], dt)



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

