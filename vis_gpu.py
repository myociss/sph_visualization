import math
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, float32
import os


@cuda.jit('void(float32[:,:], float32, float32, float32, float32[:,:])')
def mb_monotone(points, mb_threshold, mb_radius, view_size, image):
    xpix, ypix = cuda.grid(2)

    x = float32(view_size * (- 0.5 * image.shape[0] + xpix) / image.shape[0])
    y = float32(view_size * (- 0.5 * image.shape[1] + ypix) / image.shape[1])
    
    mb_sum = 0.0
    for point in points:
        #print(point.shape)
        #exit()
        dx, dy = point[0] - x, point[1] - y
        point_dist_squared = dx*dx + dy*dy
        mb_sum += mb_radius / math.sqrt(point_dist_squared)

    if mb_sum >= mb_threshold:
        image[xpix, ypix] = 1.0
    else:
        image[xpix, ypix] = 0.0

@cuda.jit('void(float32[:,:], float32, float32, float32, float32, int64, float32, float32[:,:,:])')
def mb_3d(points, mb_radius, xstart, ystart, zstart, zidx, voxel_size, image):
    xpix, ypix = cuda.grid(2)

    x = float32(xstart + voxel_size * (xpix + 0.5))
    y = float32(ystart + voxel_size * (ypix + 0.5))
    z = float32(zstart + voxel_size * (zidx + 0.5))

    mb_sum = 0.0

    has_point=0.0
    
    for point in points:
        dx, dy, dz = point[0] - x, point[1] - y, point[2] - z
        point_dist_squared = dx*dx + dy*dy + dz*dz
        mb_sum += mb_radius / math.sqrt(point_dist_squared)

        if math.sqrt(point_dist_squared) < voxel_size:
            has_point = 1.0

    image[xpix,ypix,0] = mb_sum
    image[xpix,ypix,1] = has_point

    #if mb_sum >= mb_threshold:
    #    image[xpix, ypix] = 1.0
    #else:
    #    image[xpix, ypix] = 0.0
        


if __name__ == '__main__':
    np_path = os.path.join(os.path.dirname(__file__), f'data/collision_test_pos.npy')
    with open(np_path,'rb') as f:
        pos = np.load(f)

    print(pos.shape)
    #exit()
    particle_dim = pos.shape[0]
    pos_bigplanet = pos[:int(3 * particle_dim / 4)]
    pos_smallplanet = pos[int(3 * particle_dim / 4):]

    x_vals = pos[:,:,0]
    y_vals = pos[:,:,1]
    z_vals = pos[:,:,2]

    xmin = np.min(x_vals)
    xmax = np.max(x_vals)
    ymin = np.min(y_vals)
    ymax = np.max(y_vals)
    zmin = np.min(z_vals)
    zmax = np.max(z_vals)

    img_range = max([xmax - xmin, ymax - ymin, zmax - zmin]) * 1.05
    print(img_range)

    xstart = 0.5*(xmin+xmax - img_range)
    ystart = 0.5*(ymin+ymax - img_range)
    zstart = 0.5*(zmin+zmax - img_range)

    img_dim = 256
    threads = 16
    tpb = (threads, threads)
    bpg = ( int(img_dim / threads), int(img_dim / threads) )

    voxel_size = img_range / img_dim

    all_images = np.zeros((img_dim, img_dim, img_dim), dtype='f4')

    
    xyz_string = ''
    for idx, point in enumerate(pos_bigplanet.reshape((pos_bigplanet.shape[0]*pos_bigplanet.shape[1],pos_bigplanet.shape[2]))):
        point_x = (point[0] - xstart) / img_range
        point_y = (point[1] - ystart) / img_range
        point_z = (point[2] - zstart) / img_range

        xyz_string += str(point_x) + ' ' + str(point_y) + ' ' + str(point_z)

        if idx != pos_bigplanet.shape[0]*pos_bigplanet.shape[1] - 1:
            xyz_string += '\n'

    with open('planet_collision_points_80.xyz', 'w') as f:
        f.write(xyz_string)

    exit()
    


    points_bigplanet = cuda.to_device(pos_bigplanet.reshape((pos_bigplanet.shape[0]*pos_bigplanet.shape[1],pos_bigplanet.shape[2])))
    points_smallplanet = cuda.to_device(pos_smallplanet.reshape((pos_smallplanet.shape[0]*pos_smallplanet.shape[1],pos_smallplanet.shape[2])))

    print(points_bigplanet.shape)
    print(points_smallplanet.shape)

    mb_radius_big = 0.0005
    mb_threshold_big = 3e-7

    mb_radius_small = 0.01
    mb_threshold_small = 3e-9

    bigplanet_images = np.zeros((img_dim, img_dim, img_dim), dtype='f4')
    smallplanet_images = np.zeros((img_dim, img_dim, img_dim), dtype='f4')

    center_idx = 2*img_dim // 3

    point_sum = 0.0

    all_points = np.zeros((img_dim, img_dim))

    max_proj = np.zeros(all_points.shape)

    for z_idx in range(img_dim):
        image = cuda.to_device(np.zeros((img_dim, img_dim,2), dtype='f4'))

        print(z_idx)
        mb_3d[bpg, tpb](points_bigplanet, mb_radius_big, xstart, ystart, zstart, z_idx, voxel_size, image)
        big_planet_image = image.copy_to_host()[:,:,0]
        big_planet_points = image.copy_to_host()[:,:,1]

        print(np.sum(big_planet_points))
        point_sum += np.sum(big_planet_points)

        all_points += big_planet_points

        bigplanet_images[:,:,z_idx] = big_planet_image
        
        max_proj = np.maximum(max_proj, big_planet_image)
        
        if z_idx > center_idx - 2 and z_idx < center_idx + 2:
            plt.imshow(big_planet_image)
            plt.show()

            plt.imshow(big_planet_points)
            plt.show()


        '''
        if z_idx == img_dim // 2:
            plt.imshow(big_planet_image)
            plt.show()

        mb_3d[bpg, tpb](points_smallplanet, mb_radius_small, xstart, ystart, zstart, z_idx, voxel_size, image)
        small_planet_image = image.copy_to_host()

        print(np.max(big_planet_image))
        print(np.max(small_planet_image))

        big_planet_image[big_planet_image < mb_threshold_big] = 0.0
        small_planet_image[small_planet_image < mb_threshold_small] = 0.0

        big_planet_image[big_planet_image > 0.0] = 1.0
        small_planet_image[small_planet_image > 0.0] = 2.0

        bigplanet_images[:,:,z_idx] = big_planet_image
        smallplanet_images[:,:,z_idx] = small_planet_image
        '''

        '''
        big_planet_image[small_planet_image > big_planet_image] = 0.0
        small_planet_image[big_planet_image > small_planet_image] = 0.0

        big_planet_image[big_planet_image > 0.0] = 1.0
        small_planet_image[small_planet_image > 0.0] = 2.0

        final_image = big_planet_image + small_planet_image
        print(np.max(final_image))
        '''


        
        
        #all_images[:,:,z_idx] = final_image

    #ax = plt.figure().add_subplot(projection='3d')
    #ax.voxels(bigplanet_images, edgecolor='k')
    #plt.show()

    print(point_sum)
    plt.imshow(all_points)
    plt.show()

    plt.imshow(max_proj)
    plt.show()

    '''
    plt.imshow(np.max(bigplanet_images, axis=0), cmap='magma')
    plt.show()

    plt.imshow(np.max(smallplanet_images, axis=0), cmap='magma')
    plt.show()

    plt.imshow(np.max(bigplanet_images, axis=1), cmap='magma')
    plt.show()

    plt.imshow(np.max(smallplanet_images, axis=1), cmap='magma')
    plt.show()

    plt.imshow(np.max(bigplanet_images, axis=2), cmap='magma')
    plt.show()

    plt.imshow(np.max(smallplanet_images, axis=2), cmap='magma')
    plt.show()
    '''