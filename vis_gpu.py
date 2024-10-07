import math
import numpy as np
from numba import cuda, float32


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

