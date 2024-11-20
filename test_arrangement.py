import numpy as np
import matplotlib.pyplot as plt

R = 6.99e9

G = 6.674e-8
K = 2.6e12
alpha = np.sqrt(K/2./np.pi/G)
Nbin = 64#50
N = 64*64 - 1
dxi = np.pi/Nbin

x_vals = []
y_vals = []
z_vals = []

for i in range(Nbin):
    xi1 = dxi*i
    xi2 = dxi * (i+1)
    N_in_bin =  round((np.sin(xi2)-xi2*np.cos(xi2)-np.sin(xi1)+xi1*np.cos(xi1))/np.pi*N)

    for j in range(N_in_bin):
        phi = np.random.uniform(0,2.*np.pi)
        cos_theta = np.random.uniform(-1, 1)
        xi = np.random.uniform(xi1, xi2)
        x = alpha * xi * np.sin(np.arccos(cos_theta))*np.cos(phi) #- 9.486838571279353e10
        y = alpha * xi * np.sin(np.arccos(cos_theta))*np.sin(phi) #- 5.169034544361552e9
        z = alpha * xi * cos_theta

        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)

print(np.min(x_vals))
print(np.max(x_vals))
print(np.min(y_vals))
print(np.max(y_vals))

print(len(x_vals))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter( x_vals, y_vals, c=[1 for i in range(len(x_vals))], cmap=plt.cm.jet, s=10, alpha=0.5)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter( y_vals, z_vals, c=[1 for i in range(len(x_vals))], cmap=plt.cm.jet, s=10, alpha=0.5)
plt.show()