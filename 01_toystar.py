import math
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from scipy.special import gamma
from gpu_core import calc_dv_toystar, calc_density, update_pos_vel_halfstep
from smoothing_length import calc_h_guesses, calc_midpoint, calc_zeta, bisect_update, get_new_smoothing_lengths
from pmocz_functions import plot_frame


def getPairwiseSeparations( ri, rj ):
    """
    Get pairwise desprations between 2 sets of coordinates
    ri    is an M x 3 matrix of positions
    rj    is an N x 3 matrix of positions
    dx, dy, dz   are M x N matrices of separations
    """
    
    M = ri.shape[0]
    N = rj.shape[0]
    
    # positions ri = (x,y,z)
    rix = ri[:,0].reshape((M,1))
    riy = ri[:,1].reshape((M,1))
    riz = ri[:,2].reshape((M,1))
    
    # other set of points positions rj = (x,y,z)
    rjx = rj[:,0].reshape((N,1))
    rjy = rj[:,1].reshape((N,1))
    rjz = rj[:,2].reshape((N,1))
    
    # matrices that store all pairwise particle separations: r_i - r_j
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T
    
    return dx, dy, dz

def W( x, y, z, h ):
    """
    Gausssian Smoothing kernel (3D)
    x     is a vector/matrix of x positions
    y     is a vector/matrix of y positions
    z     is a vector/matrix of z positions
    h     is the smoothing length
    w     is the evaluated smoothing function
    """
    
    r = np.sqrt(x**2 + y**2 + z**2)
    
    w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
    
    return w

def getDensity( r, pos, m, h ):
    """
    Get Density at sampling loctions from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of accelerations
    """
    
    M = r.shape[0]
    
    dx, dy, dz = getPairwiseSeparations( r, pos )
    
    rho = np.sum( m * W(dx, dy, dz, h), 1 ).reshape((M,1))
    
    return rho

'''
# cuda stuff
particle_dim = 20#32
threads = 16
tpb = (threads, threads)
bpg = ( int(particle_dim / threads), int(particle_dim / threads) )
N = particle_dim * particle_dim

# astrophysics stuff


#R = 7000000 # white dwarf radius
#solar_mass = 1.989e30
#M=0.6*solar_mass # white dwarf mass

#eq_state_const = 0.1 * (R * 4/3)   # equation of state constant
#polytropic_idx = 3/2 # polytropic index
#viscosity = 1 # damping


M = 2 # star mass
R = 0.75 # star radius

h_init = 0.1


eq_state_const = 0.1 * (R * 4/3)
polytropic_idx = 1
viscosity = 1

particle_mass = M/N

lmbda_2d = 2*eq_state_const*np.pi**(-1/polytropic_idx) * ( ( (M*(1+polytropic_idx)) / (R**2) )**(1 + 1/polytropic_idx) ) / M
lmbda_3d = 2*eq_state_const*(1+polytropic_idx)*np.pi**(-3/(2*polytropic_idx)) * (M*gamma(5/2+polytropic_idx)/R**3/gamma(1+polytropic_idx))**(1/polytropic_idx) / R**2


# simulation stuff
t = 0
tEnd = 12
dt = 0.04#0.005

kernel_radius = 3.0

for d in [3]:
    init_pos = (R * 4/3) * np.random.randn(particle_dim, particle_dim, 2, d).astype('f4')

    d_pos = cuda.to_device(init_pos)
    d_vel = cuda.to_device(np.zeros((particle_dim, particle_dim, 2, d), dtype='f4'))
    d_dV = cuda.to_device(np.zeros((particle_dim, particle_dim, d), dtype='f4'))
    d_rho = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))
    #d_smoothing = cuda.to_device(np.zeros((particle_dim, particle_dim), dtype='f4'))

    x = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))
    y = cuda.to_device(np.zeros((particle_dim, particle_dim, 3), dtype='f4'))

    lmbda = lmbda_2d if d == 2 else lmbda_3d

    print(d_pos.shape)
    print(x.shape)
    print(y.shape)

    #get_new_smoothing_lengths(d_pos[:,:,0,:], x, y, particle_mass, kernel_radius, tpb, bpg)

    steps = int(tEnd/dt)

    smoothing_lengths = np.zeros((particle_dim, particle_dim)) + h_init
    smoothing_lengths = cuda.to_device(smoothing_lengths.astype('f4'))
    #steps = 500
    for i in range(1, steps):
        print(i)

        update_pos_vel_halfstep[bpg, tpb](d_pos, d_vel, d_dV, dt)
    
        #get_new_smoothing_lengths(d_pos[:,:,0,:], x, y, particle_mass, kernel_radius, tpb, bpg, n_iter=20)
        #smoothing_lengths = x[:,:,1]

        calc_density[bpg, tpb](d_pos[:,:,0,:], particle_mass, smoothing_lengths, d_rho)

        print('----------------------------------------')
        print(np.max(d_rho.copy_to_host()))
        print(np.min(d_rho.copy_to_host()))

        print(np.max(smoothing_lengths.copy_to_host()))
        print(np.min(smoothing_lengths.copy_to_host()))
    

        calc_dv_toystar[bpg, tpb](d_pos[:,:,0,:], d_vel[:,:,0,:], particle_mass, smoothing_lengths, eq_state_const, polytropic_idx, lmbda, viscosity, d_rho, d_dV)
        update_pos_vel_halfstep[bpg, tpb](d_pos, d_vel, d_dV, dt)

        #for i1 in range(particle_dim):
        #    for j1 in range(particle_dim):
        #        print('...')
        #        print(d_pos.copy_to_host()[i1,j1,0])
        #        print(d_dV.copy_to_host()[i1,j1])

        #if i == 2:
        #    exit()

    plot_vals = np.reshape(d_pos[:,:,0,:].copy_to_host(), (particle_dim*particle_dim,d))#[:,:2] #d_pos.copy_to_host()[:,:,0][:2]

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor('black')

    cval = d_rho.copy_to_host()#np.minimum((rho.copy_to_host()-3)/3,1).flatten()
    ax1.scatter(plot_vals[:,0],plot_vals[:,1], c=cval, cmap=plt.cm.jet, s=10, alpha=0.5)
    plt.show()
'''

dim = 32
threads = 16
tpb = (threads, threads)
bpg = ( int(dim / threads), int(dim / threads) )

d_rho = cuda.to_device(np.zeros((dim, dim), dtype='f4'))
d_vel = cuda.to_device(np.zeros((dim, dim, 2, 3), dtype='f4'))
d_dV = cuda.to_device(np.zeros((dim, dim, 3), dtype='f4'))

d_density_smoothing = cuda.to_device(np.zeros((dim, dim, 3), dtype='f4'))



N         = dim*dim    # Number of particles
t         = 0      # current time of the simulation
tEnd      = 1#12     # time at which simulation ends
#dt        = 0.01   # timestep
dt = 0.01


#R = 7000000 # white dwarf radius
#solar_mass = 1.989e30
#M=0.6*solar_mass # white dwarf mass
#polytropic_idx = 3/2#1      # polytropic index

R = 0.75 # star radius
M = 2 # star mass
polytropic_idx = 3/2


h_init = 0.1 * (R * 4/3) #0.1 * (R * 4/3)    # smoothing length
eq_state_const = 0.1 * (R * 4/3)   # equation of state constant
viscosity = 1      # damping


smoothing_length = np.zeros((dim, dim)) + h_init
smoothing_length = cuda.to_device(smoothing_length.astype('f4'))

init_pos = (R * 4/3) * np.random.randn(dim, dim, 2, 3).astype('f4')
d_pos = cuda.to_device(init_pos)
'''
M         = 2     # star mass
R         = 0.9 #0.75   # star radius
smoothing_length = 0.1    # smoothing length
eq_state_const = 0.1   # equation of state constant
polytropic_idx = 3/2#1      # polytropic index
viscosity = 1      # damping
'''

particle_mass = M/N

lmbda = 2*eq_state_const*(1+polytropic_idx)*np.pi**(-3/(2*polytropic_idx)) * (M*gamma(5/2+polytropic_idx)/R**3/gamma(1+polytropic_idx))**(1/polytropic_idx) / R**2

print(lmbda)

rr = np.zeros((100,3))
rlin = np.linspace(0,R,100)
rr[:,0] =rlin

rho_analytic = (lmbda * (R**2 - rlin**2)/(2*eq_state_const*(1+polytropic_idx)))**polytropic_idx  #lmbda/(4*eq_state_const) * (R**2 - rlin**2)


print(rho_analytic)

plt.plot(rho_analytic)
plt.show()

#exit()


#S = np.zeros((N, 7) , dtype='f4')

#S[:3] = np.random.randn(N, 3)

#S_int = RK45(integrate, t, S, tEnd, max_step = dt)

steps = int(tEnd/dt)

all_pos = np.zeros((N, 3, steps), dtype='f4')

all_pos[:,:, 0] = init_pos[:,:,0,:].reshape((N, 3))

x = cuda.to_device(np.zeros((dim, dim, 3), dtype='f4'))
y = cuda.to_device(np.zeros((dim, dim, 3), dtype='f4'))

#while t < tEnd:
for i in range(1, steps):
    print(i)
    #pos_i = d_pos.copy_to_host()
    #vel_i = d_vel.copy_to_host()
    #dV_i = d_dV.copy_to_host()

    update_pos_vel_halfstep[bpg, tpb](d_pos, d_vel, d_dV, dt)
    calc_density[bpg, tpb](d_pos[:,:,0,:], particle_mass, smoothing_length, d_rho)

    get_new_smoothing_lengths(d_pos[:,:,0,:], x, y, particle_mass, 3.0, tpb, bpg, n_iter=20)
    smoothing_length = x[:,:,1]
    smoothing_length_y_cpu = y[:,:,1].copy_to_host()
    assert np.all(np.abs(smoothing_length_y_cpu) < 0.001 * R)
    

    calc_dv_toystar[bpg, tpb](d_pos[:,:,0,:], d_vel[:,:,0,:], particle_mass, smoothing_length, eq_state_const, polytropic_idx, lmbda, viscosity, d_rho, d_dV)

    update_pos_vel_halfstep[bpg, tpb](d_pos, d_vel, d_dV, dt)

    all_pos[:,:, i] = (d_pos.copy_to_host()[:,:,0,:]).reshape((N, 3))






#density = d_density_smoothing.copy_to_host()[:,:,0]
density = d_rho.copy_to_host()


pos = all_pos[:,:,-1]
plot_frame(pos, density, R, polytropic_idx, eq_state_const, h_init, particle_mass, lmbda, 'toystar_3d.png')

'''
x_pos = pos[0,0,0]
y_pos = pos[0,0,1]
z_pos = pos[0,0,2]

rho = 0.0

omega = 0.0

smoothing_init = smoothing_length

for i1 in range(pos.shape[0]):
    for j1 in range(pos.shape[1]):
        #if i == i1 and j == j1:
        #    continue

        x_dist = x_pos - pos[i1, j1, 0, 0]
        y_dist = y_pos - pos[i1, j1, 0, 1]
        z_dist = z_pos - pos[i1, j1, 0, 2]

        r = math.sqrt(x_dist*x_dist + y_dist*y_dist + z_dist*z_dist)

        w = kernel_w_cubic(r, smoothing_init)
        rho += particle_mass * w

        omega += particle_mass * ( ( (-r / smoothing_init) * kernel_grad_w_cubic(r, smoothing_init) )  -  ( 3.0 * w / smoothing_init ) )

h1 = eta_const / smoothing_init

zeta_hj = particle_mass * (h1**dim) - rho
'''


'''
cval = np.minimum((density-3)/3,1).flatten()

plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.5)

plt.show()
'''


fig = plt.figure(figsize=(4,5), dpi=80)
grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
ax1 = plt.subplot(grid[0:2,0])
ax2 = plt.subplot(grid[2,0])
#exit()



plt.sca(ax1)
plt.cla()

#cval = np.minimum((density-3*(R*4/3))/3,1).flatten()
cval = density

plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.jet, s=10, alpha=0.5)
ax1.set(xlim=(-1.4 * (R * 4/3), 1.4 * (R * 4/3)), ylim=(-1.2 * (R * 4/3), 1.2 * (R * 4/3)))
ax1.set_aspect('equal', 'box')
ax1.set_xticks([-(R * 4/3),0,(R * 4/3)])
ax1.set_yticks([-(R * 4/3),0,(R * 4/3)])
ax1.set_facecolor('black')
ax1.set_facecolor((.1,.1,.1))
            
plt.sca(ax2)
plt.cla()
#ax2.set(xlim=(0, R), ylim=(0, 3e9))
#ax2.set_aspect(0.1)
plt.plot(rlin, rho_analytic, color='gray', linewidth=2)
rho_radial = getDensity( rr, pos, particle_mass, h_init )
plt.plot(rlin, rho_radial, color='blue')

print(rho_analytic)

rho_radial = [round(elem[0]) for elem in rho_radial.tolist()]
print(rho_radial)

error = sum([(rho_analytic[i] - rho_radial[i])**2 for i in range(len(rho_radial))]) / len(rho_radial)

print(error/R)

plt.show()
