# all functions in this file are taken or adapted from https://philip-mocz.medium.com/create-your-own-smoothed-particle-hydrodynamics-simulation-with-python-76e1cec505f1
# code at https://github.com/pmocz/sph-python

import math
import numpy as np
import matplotlib.pyplot as plt
from kernels import w_gauss, dwdq_gauss

def plot_frame(pos, dens, R, polytropic_idx, eq_state_const, h, particle_mass, lmbda, fig_name):
    rr = np.zeros((100,3))
    rlin = np.linspace(0,R,100)
    rr[:,0] =rlin

    rho_analytic = (lmbda * (R**2 - rlin**2)/(2*eq_state_const*(1+polytropic_idx)))**polytropic_idx

    fig = plt.figure(figsize=(4,5), dpi=80)
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2,0])
    ax2 = plt.subplot(grid[2,0])
    #exit()



    plt.sca(ax1)
    plt.cla()

    #cval = np.minimum((density-3*(R*4/3))/3,1).flatten()
    cval = dens

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
    rho_radial = density( pos, h, particle_mass, pos.shape[1], rr=rr )
    plt.plot(rlin, rho_radial, color='blue')
    plt.savefig(fig_name)
    plt.show()


def density( pos, h, particle_mass, dim, rr=None ):
    """
    Get Density at sampling loctions from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of densities
    """

    if rr is None:
        rr = pos
	
    M = rr.shape[0]
	
    x_dist, y_dist, z_dist = pairwise_separations( rr, pos )
    radius = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2).astype('f4')
	
    rho = np.sum( particle_mass * w_gauss(radius, h, dim), 1 ).reshape((M,1))
	
    return rho

def pressure(rho, eq_state_const, polytropic_idx):
	"""
	Equation of State
	rho   vector of densities
	k     equation of state constant
	n     polytropic index
	P     pressure
	"""
	
	P = eq_state_const * rho**(1+1/polytropic_idx)
	
	return P

def dv( pos, vel, h, eq_state_const, polytropic_idx, lmbda, viscosity, particle_mass, dim ):
    """
    Calculate the acceleration on each SPH particle
    pos   is an N x 3 matrix of positions
    vel   is an N x 3 matrix of velocities
    m     is the particle mass
    h     is the smoothing length
    k     equation of state constant
    n     polytropic index
    lmbda external force constant
    nu    viscosity
    a     is N x 3 matrix of accelerations
    """
    
    N = pos.shape[0]
    
    # Calculate densities at the position of the particles
    rho = density( pos, h, particle_mass, dim )
    
    # Get the pressures
    P = pressure(rho, eq_state_const, polytropic_idx)

    # Get pairwise distances and gradients
    x_dist, y_dist, z_dist = pairwise_separations( pos, pos )
    radius = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2).astype('f4')
    
    #dWx, dWy, dWz = gradW( dx, dy, dz, h )
    dwdq = dwdq_gauss(radius, h, dim)

    grad_w = np.zeros(radius.shape).astype('f4')

    np.putmask(grad_w, radius > 1e-12, dwdq * (1./h) / radius )

    dWx = x_dist * grad_w
    dWy = y_dist * grad_w
    dWz = z_dist * grad_w
    
    # Add Pressure contribution to accelerations
    ax = - np.sum( particle_mass * ( P/rho**2 + P.T/rho.T**2  ) * dWx, 1).reshape((N,1))
    ay = - np.sum( particle_mass * ( P/rho**2 + P.T/rho.T**2  ) * dWy, 1).reshape((N,1))

    if dim == 3:
        az = - np.sum( particle_mass * ( P/rho**2 + P.T/rho.T**2  ) * dWz, 1).reshape((N,1))
        a = np.hstack((ax,ay,az))
    else:
        a = np.hstack((ax,ay))
    
    # Add external potential force
    a -= lmbda * pos[:,:dim]
    
    # Add viscosity
    a -= viscosity * vel[:,:dim]
    
    return a


def pairwise_separations( ri, rj ):
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

    #if ri.shape[1] == 3:
	riz = ri[:,2].reshape((M,1)) if ri.shape[1] == 3 else None
	
	# other set of points positions rj = (x,y,z)
	rjx = rj[:,0].reshape((N,1))
	rjy = rj[:,1].reshape((N,1))
	rjz = rj[:,2].reshape((N,1)) if ri.shape[1] == 3 else None
	
	# matrices that store all pairwise particle separations: r_i - r_j
	dx = rix - rjx.T
	dy = riy - rjy.T
	dz = riz - rjz.T if ri.shape[1] == 3 else np.zeros(dx.shape)
	
	return dx, dy, dz