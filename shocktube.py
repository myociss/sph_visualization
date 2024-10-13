import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from kernels import w_gauss, dwdq_gauss, w_quintic, dwdq_quintic
from astro_core import calc_density, calc_pressure_from_energy, calc_params_shocktube
from numba import cuda


# Function to find the roots of!
def f(P, pL, pR, cL, cR, gg):
    a = (gg-1)*(cR/cL)*(P-1) 
    b = np.sqrt( 2*gg*(2*gg + (gg+1)*(P-1) ) )
    return P - pL/pR*( 1 - a/b )**(2.*gg/(gg-1.))

# Analtyic Sol to Sod Shock
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

@np.vectorize 
def cubic_spline(r, h, order):
    
    q = float(abs(r))/h
    sigma = float(2.0 / (3.0 * h))
    if order == 0:
        if q <=1.0 and q >=0.0: 
            return sigma * (1.0 - (1.5 * q * q) + (0.75 * q * q * q))
        elif q > 1.0 and q <= 2.0:
            return sigma * 0.25 * ((2.0 - q) ** 3.0)
        else:
            return 0.0
    else:
        diff_multiplier = float(np.sign(r) / h)
        if q <=1.0 and q >=0.0: 
            return float(sigma * ((-3.0 * q) + (2.25 * q * q)) * diff_multiplier)
        elif q > 1.0 and q <= 2.0:
            return float(sigma * -0.75 * ((2 - q) ** 2) * diff_multiplier)
        else:
            return 0.0

class Shocktube(object):
    
    ### Need to add solid boundaries ? ###
    
    def __init__(self, x, p, rho, v, m, h, gamma, epsilon, eta, kernel):
        
        self.x = x
        self.p = p
        self.rho = rho
        self.e = p / ((gamma - 1) * rho)
        self.m = m
        self.h = h
        self.gamma = gamma
        self.epsilon = epsilon
        self.eta = eta
        self.kernel = kernel
        self.v = v
        self.num = len(self.x)
        
    #### One Step Euler Integrator #### 

    def update_euler(self, dt):
        
        ## Define temp arrays for storing increments ##
        
        x_inc = np.zeros_like(self.x, dtype='float32')
        rho_inc = np.zeros_like(self.x, dtype='float32')
        v_inc = np.zeros_like(self.x, dtype='float32')
        e_inc = np.zeros_like(self.x, dtype='float32')
        
        ## Iterate over all particles and store the accelerations in temp arrays ##
        ## 10 particles to the left and 10 particles to the right as boundary ###
        for i in range(35, self.num - 35):

            ### Evaluating variables ###

            vij = self.v[i] - self.v
            dwij = self.kernel(self.x[i] - self.x, self.h, order = 1)
            wij = self.kernel(self.x[i] - self.x, self.h, order = 0)
            xij = self.x[i] - self.x

            p_rho_i = self.p[i] / (self.rho[i] * self.rho[i])
            p_rho_j = self.p / (self.rho * self.rho)
            
            p_rho_ij =  p_rho_i + p_rho_j

            ### Artificial viscosity ###

            numerator = self.h * vij * xij
            denominator = (xij ** 2.0) + (self.eta ** 2.0)
            mu_ij = numerator / denominator
            mu_ij[mu_ij > 0] = 0.0 #only activated for approaching particles

            ci = (self.gamma * self.p[i] / self.rho[i]) ** 0.5
            cj = (self.gamma * self.p / self.rho) ** 0.5
            cij = 0.5 * (ci + cj)
            rhoij = 0.5 * (self.rho[i] + self.rho)
            numerator = (-1 * cij * mu_ij) + (mu_ij ** 2)
            denominator = rhoij
            pi_ij = numerator / denominator

            ### Evaluating gradients ###

            grad_rho = self.rho[i] * np.sum(self.m * vij * dwij / self.rho)
            grad_v = -1 * np.sum(self.m * (p_rho_ij + pi_ij) * dwij)            
            grad_e = 0.5 * np.sum(self.m * (p_rho_ij + pi_ij) * vij * dwij)

            rho_inc[i] = dt * grad_rho
            v_inc[i] = dt * grad_v
            e_inc[i] = dt * grad_e
            
            ### Get XSPH Velocity and calculate increment ###

            rho_avg = 0.5 * (self.rho[i] + self.rho)
            correction = self.epsilon * np.sum(self.m * -1 * vij * wij / rho_avg)
            xsph_velocity =  self.v[i] + correction

            x_inc[i] = dt * xsph_velocity

            
        ## Update the original arrays using the increment arrays ##
            
        self.rho += rho_inc
        self.v += v_inc
        self.e += e_inc

        ### Update pressure ###
        
        self.p = (self.gamma - 1) * self.e * self.rho

        ### Update positions ###

        self.x += x_inc

    def update_rho(self, particle_dim):
        threads = 16
        tpb = (threads, threads)
        bpg = ( int(particle_dim / threads), int(particle_dim / threads) )

        d_pos = cuda.to_device(np.reshape(self.x, (particle_dim, particle_dim, 1)).astype('f4'))
        d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
        d_e = cuda.to_device(np.reshape(self.e, (particle_dim, particle_dim)).astype('f4'))
        d_pressure = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))

        smoothing_length = np.zeros((particle_dim, particle_dim)) + self.h
        smoothing_length = cuda.to_device(smoothing_length.astype('f4'))

        calc_density[bpg, tpb](d_pos, self.m, smoothing_length, d_rho)

        self.rho = np.reshape(d_rho.copy_to_host().astype('f8'), self.x.shape)

        #print(d_pressure.shape)
        #print(d_e.shape)
        #print(d_rho.shape)
        #print(self.gamma)

        calc_pressure_from_energy[bpg, tpb](d_rho, d_e, self.gamma, d_pressure)

        self.p = np.reshape(d_pressure.copy_to_host().astype('f8'), self.x.shape)

    def get_dv_de(self, dt, mask):
        threads = 16
        tpb = (threads, threads)
        bpg = ( int(particle_dim / threads), int(particle_dim / threads) )

        d_pos = cuda.to_device(np.reshape(self.x, (particle_dim, particle_dim, 1)).astype('f4'))
        d_vel = cuda.to_device(np.reshape(self.v, (particle_dim, particle_dim, 1)).astype('f4'))
        d_rho = cuda.to_device(np.reshape(self.rho, (particle_dim, particle_dim)).astype('f4'))
        d_pressure = cuda.to_device(np.reshape(self.p, (particle_dim, particle_dim)).astype('f4'))

        d_de = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
        d_dv = cuda.to_device(np.zeros((particle_dim, particle_dim, 1), dtype='f4'))
        d_mask = cuda.to_device(np.reshape(mask, (particle_dim, particle_dim)).astype('f4'))

        #d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
        #d_pressure = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))

        smoothing_length = np.zeros((particle_dim, particle_dim)) + self.h
        smoothing_length = cuda.to_device(smoothing_length.astype('f4'))

        calc_params_shocktube[bpg, tpb](d_pos, d_vel, self.m, smoothing_length, self.gamma, d_rho, d_pressure, d_dv, d_de, d_mask)

        dv = np.reshape(d_dv.copy_to_host().astype('f8'), self.x.shape)
        de = np.reshape(d_de.copy_to_host().astype('f8'), self.x.shape)

        print(np.max(dv))
        print(np.min(dv))
        print(np.max(de))
        print(np.min(de))

        return dv, de



    def get_dv_de_save(self, dt, mask):
        v_inc = np.zeros_like(self.x, dtype='float32')
        e_inc = np.zeros_like(self.x, dtype='float32')

        alpha_visc = 1
        beta_visc = 2
        epsilon_visc = 0.01

        x_inc = np.zeros_like(self.x, dtype='float32')


        for i in range(self.num):
            vij = self.v[i] - self.v
            xij = self.x[i] - self.x

            wij = w_gauss(np.abs(xij), self.h, 1)
            
            dwij = xij * dwdq_gauss(np.abs(xij), self.h, 1) * (1./self.h) / np.abs(xij)
            dwij = np.nan_to_num(dwij)


            p_rho_i = self.p[i] / (self.rho[i] * self.rho[i])
            p_rho_j = self.p / (self.rho * self.rho)
            
            p_rho_ij =  p_rho_i + p_rho_j

            numerator = self.h * vij * xij
            denominator = (xij ** 2.0) + (epsilon_visc * self.h * self.h)
            mu_ij = numerator / denominator
            mu_ij[mu_ij > 0] = 0.0

            ci = (self.gamma * self.p[i] / self.rho[i]) ** 0.5
            cj = (self.gamma * self.p / self.rho) ** 0.5
            cij = 0.5 * (ci + cj)
            rhoij = 0.5 * (self.rho[i] + self.rho)

            numerator = (-alpha_visc * cij * mu_ij) + beta_visc * (mu_ij ** 2)
            denominator = rhoij
            pi_ij = numerator / denominator

            grad_v = -1 * np.sum(self.m * (p_rho_ij + pi_ij) * dwij)            
            grad_e = 0.5 * np.sum(self.m * (p_rho_ij + pi_ij) * vij * dwij)

            v_inc[i] = grad_v#dt * grad_v
            e_inc[i] = grad_e#dt * grad_e
            
            ### Get XSPH Velocity and calculate increment ###
            #rho_avg = 0.5 * (self.rho[i] + self.rho)
            #correction = self.epsilon * np.sum(self.m * -1 * vij * wij / rho_avg)
            #xsph_velocity =  self.v[i] + correction

            #x_inc[i] = dt * xsph_velocity

        #self.v += v_inc * mask
        #self.e += e_inc * mask

        ### Update positions ###

        #self.x += x_inc
        return v_inc * mask, e_inc * mask



    def update_euler_SD(self, dt, mask):
        
        ## Define temp arrays for storing increments ##
        
        x_inc = np.zeros_like(self.x, dtype='float32')
        rho_inc = np.copy(self.rho) # not increment, direct corection
        v_inc = np.zeros_like(self.x, dtype='float32')
        e_inc = np.zeros_like(self.x, dtype='float32')
        
        ## Iterate over all particles and store the accelerations in temp arrays ##
        ## 10 particles to the left and 10 particles to the right as boundary ###
        #for i in range(35, self.num - 35):
        for i in range(self.num):

            ### Evaluating variables ###

            vij = self.v[i] - self.v
            #dwij = self.kernel(self.x[i] - self.x, self.h, order = 1)
            #wij = self.kernel(self.x[i] - self.x, self.h, order = 0)
            xij = self.x[i] - self.x

            #wij = np.array([w_quintic(abs(elem), self.h, 1) for elem in xij])
            #dwij = dwdq_gauss(xij, self.h, 1) * (1./self.h) / xij
            wij = w_gauss(np.abs(xij), self.h, 1)
            
            dwij = xij * dwdq_gauss(np.abs(xij), self.h, 1) * (1./self.h) / np.abs(xij)
            dwij = np.nan_to_num(dwij)

            #dwij = dwdq_gauss(np.abs(xij), self.h, 1) * (1./self.h) / np.abs(xij)
            #np.nan_to_num(dwij)
            #dwij *= xij

            p_rho_i = self.p[i] / (self.rho[i] * self.rho[i])
            p_rho_j = self.p / (self.rho * self.rho)
            
            p_rho_ij =  p_rho_i + p_rho_j

            ### Artificial viscosity ###

            numerator = self.h * vij * xij
            denominator = (xij ** 2.0) + (self.eta ** 2.0)
            mu_ij = numerator / denominator
            mu_ij[mu_ij > 0] = 0.0 #only activated for approaching particles

            ci = (self.gamma * self.p[i] / self.rho[i]) ** 0.5
            cj = (self.gamma * self.p / self.rho) ** 0.5
            cij = 0.5 * (ci + cj)
            rhoij = 0.5 * (self.rho[i] + self.rho)
            numerator = (-1 * cij * mu_ij) + (mu_ij ** 2)
            denominator = rhoij
            pi_ij = numerator / denominator

            ### Evaluating gradients ###

            grad_v = -1 * np.sum(self.m * (p_rho_ij + pi_ij) * dwij)            
            grad_e = 0.5 * np.sum(self.m * (p_rho_ij + pi_ij) * vij * dwij)

            rho_inc[i] = np.sum(self.m * wij)
            v_inc[i] = dt * grad_v
            e_inc[i] = dt * grad_e
            
            ### Get XSPH Velocity and calculate increment ###

            rho_avg = 0.5 * (self.rho[i] + self.rho)
            correction = self.epsilon * np.sum(self.m * -1 * vij * wij / rho_avg)
            xsph_velocity =  self.v[i] + correction

            x_inc[i] = dt * xsph_velocity

            
        ## Update the original arrays using the increment arrays ##
            
        self.rho = rho_inc#not incremented, corrected directly
        self.v += v_inc * mask
        self.e += e_inc * mask

        ### Update pressure ###
        
        self.p = (self.gamma - 1) * self.e * self.rho

        ### Update positions ###

        self.x += x_inc
        
if __name__ == '__main__':

    total_mass = 0.75

    particle_dim = 32

    n_particles = particle_dim*particle_dim # - (35*2)

    n_right = n_particles // 5

    #n_particles = n_right * 5

    dx_left = 0.6 / (n_particles - n_right)
    dx_right = 0.6 / n_right
    #left = np.linspace(-0.5,0,320*2)
    #left = np.linspace(-0.5,0,n_particles - n_right)
    left = np.array([i*dx_left - 0.6 + dx_left for i in range(n_particles - n_right)])
    right = np.array([(i - (n_particles - n_right))*dx_right + 0.5*dx_right for i in range((n_particles-n_right), n_particles)])
    #dxl = left[1] - left[0]
    #right = np.linspace(0,0.5,40*2)[1:]
    #right = np.linspace(0,0.5,n_right+1)[1:]
    #dxr = right[1] - right[0]

    #left_boundary = np.linspace(-0.5 - (35 * dxl), -0.5 - dxl, 35)
    #right_boundary = np.linspace(0.5 + dxr, 0.5 + (35 * dxr), 35)

    h = 2*(right[1] - right[0])

    #left = np.append(left_boundary, left)
    #right = np.append(right, right_boundary)

    x = np.append(left, right)

    #print(left_boundary.shape)
    #print(right_boundary.shape)
    print(x.shape)
    #exit()

    #rho = np.append(np.ones_like(left), np.ones_like(right)*0.125)
    rho = np.append(np.ones_like(left), np.ones_like(right)*0.25)
    #p = np.append(np.ones_like(left), np.ones_like(right)*0.1)
    p = np.append(np.ones_like(left), np.ones_like(right)*0.1795)
    v = np.zeros_like(x)
    gamma = 1.4
    epsilon = 0.5
    eta = 1e-04
    m = total_mass / n_particles

    st = Shocktube(x = x, p = p, rho = rho, \
                   v = v, m = m, h = h, gamma = gamma, \
                   epsilon = epsilon, eta = eta, kernel = cubic_spline)

    n_boundary = 35
    mask = np.zeros(x.shape) + 1.0
    mask[:n_boundary] = 0.0
    mask[-n_boundary:] = 0.0

    #print(mask[])
    #exit()

    dv = np.zeros(x.shape)
    de = np.zeros(x.shape)

    dt = 1e-04

    steps = 500

    for i in range(steps):
        st.x += 0.5 * dt * st.v
        st.v += 0.5 * dt * dv
        st.e += 0.5 * dt * de
        #dv, de = st.get_dv_de(dt=1e-04, mask=mask)

        st.update_rho(particle_dim)
        #st.update_euler_SD(dt=1e-04, mask=mask)
        
        dv, de = st.get_dv_de(dt=dt, mask=mask)
        #dv, de = st.get_dv_de_save(dt=dt, mask=mask)
        '''
        dv_alt, de_alt = st.get_dv_de_save(dt=dt, mask=mask)

        print('...')
        print(np.max(dv_alt))
        print(np.min(dv_alt))

        print(np.max(de_alt))
        print(np.min(de_alt))
        '''

        st.x += 0.5 * dt * st.v
        st.v += 0.5 * dt * dv
        st.e += 0.5 * dt * de

        print(i)

    #plt.scatter(st.x, st.rho)
    #plt.show()

    #plt.scatter(st.x, st.v)
    #plt.show()

    indexes = np.where((x > -0.4) & (x < 0.4))
    
    Nx = 100
    X = 1.
    dx = X/(Nx-1)
    xs = np.linspace(0,0.8,Nx)
    x0 = Nx//2
    analytic = SodShockAnalytic(1.0, 0.0, 1.0, 0.25, 0.0, 0.1795, xs, x0, steps*1e-04, 1.4)


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(st.x[indexes], st.rho[indexes], marker='.')
    ax1.plot(xs - 0.4,analytic[0].T, color='r')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(st.x[indexes], st.v[indexes], marker='.')
    ax1.plot(xs - 0.4,analytic[1].T, color='r')
    plt.show()

    #plt.scatter(st.x, st.v)
    #plt.show()
    