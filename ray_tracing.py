import numpy as np
import os
import subprocess
import sys
import matplotlib.pyplot as plt
import meshio
from numba import cuda#, float64
import time
#from aabbtree import AABB
#from aabbtree import AABBTree

M_EPSILON = 0.00001

AABB_PARENT = 0
AABB_XMIN = 1
AABB_XMAX = 2
AABB_YMIN = 3
AABB_YMAX = 4
AABB_ZMIN = 5
AABB_ZMAX = 6
AABB_LEFT = 7
AABB_RIGHT = 8
AABB_DATA = 9

NODE_LEFT = 0
NODE_RIGHT = 1
NODE_DATA = 2

@cuda.jit('float64(float64, float64)', device=True)
def fix_t_gpu(t_val, ray_dir):
    if ray_dir == 0.0:
        if t_val == 0.0:
            return 0.0
        elif t_val < 0:
            return -np.inf
        else:
            return np.inf
    else:
        return (1.0 / ray_dir) * t_val


'''
#@cuda.jit('float64(float64[:,:], float64[:], float64, float64, float64, float64, float64, float64, float64[:,:,:])', device=True)
def get_nearest(aabbtree, complete_list, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, triangles):

    tree_index = 0
    min_dist = np.inf

    iterations = 0

    #print(complete_list[0])

    while complete_list[0] == 0.0:
        #iterations += 1
        node = aabbtree[tree_index]
        left_idx = int(node[AABB_LEFT])
        right_idx = int(node[AABB_RIGHT])

        if left_idx == -1 and right_idx == -1:
            min_dist = 0.0
            complete_list[tree_index] = 1.0
            tree_index = int(node[AABB_PARENT])
            continue

        if complete_list[left_idx] > 0.0 and complete_list[right_idx] > 0.0:
            complete_list[tree_index] = 1.0
            tree_index = int(node[AABB_PARENT])
            continue

        # test intersection
        bbox_intersection = node[AABB_XMIN] <= ray_origin_x <= node[AABB_XMAX] and node[AABB_YMIN] <= ray_origin_y <= node[AABB_YMAX]

        if bbox_intersection:
            if complete_list[left_idx] == 1.0:
                tree_index = right_idx
            else:
                tree_index = left_idx
        else:
            complete_list[tree_index] = 1.0
            tree_index = int(node[AABB_PARENT])

    #print(iterations)
    return min_dist
'''


#@cuda.jit('float64(float64[:,:], float64, float64, float64, float64, float64, float64, int64[:], float64[:,:,:])', device=True)
#def get_nearest(aabbtree, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, stack, triangles):
@cuda.jit('float64(float64[:,:], int64, int64, float64, float64, float64, float64, float64, float64, int64[:,:,:], float64[:,:,:])', device=True)
def get_nearest(aabbtree, xpix, ypix, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, stack, triangles):
    stack_pointer = 0

    stack[xpix,ypix,stack_pointer+1] = 0
    stack_pointer += 1

    node = aabbtree[0]

    #if (not node[AABB_DATA] > -1.0) and node[AABB_XMIN] <= ray_origin_x <= node[AABB_XMAX] and node[AABB_YMIN] <= ray_origin_y <= node[AABB_YMAX]:
    #    nearest_dist = 0.0

    nearest_dist = np.inf



    while stack_pointer > 0:
        tree_index = stack[xpix,ypix,stack_pointer - 1]
        stack_pointer -= 1

        # ?????????
        #stack_pointer -= 1

        node = aabbtree[tree_index]

        #if node[AABB_DATA] > -1.0:
        #    continue
        if node[AABB_LEFT] == -1 and node[AABB_RIGHT] == -1:
            #nearest_dist = 0.0
            #continue

            
            #nearest_dist = 0.0
            tri = triangles[int(node[AABB_DATA])]
            v0 = tri[0]
            v1 = tri[1]
            v2 = tri[2]

            # e1 = v1 - v0
            e1_x = v1[0] - v0[0]
            e1_y = v1[1] - v0[1]
            e1_z = v1[2] - v0[2]

            # e2 = v2 - v0
            e2_x = v2[0] - v0[0]
            e2_y = v2[1] - v0[1]
            e2_z = v2[2] - v0[2]

            # P = cross(ray.direction, e2)
            P_x = ray_dir_y*e2_z - ray_dir_z*e2_y
            P_y = ray_dir_z*e2_x - ray_dir_x*e2_z
            P_z = ray_dir_x*e2_y - ray_dir_y*e2_x

            # det = dot(e1, P)
            det = e1_x*P_x + e1_y*P_y + e1_z*P_z

            # if (det > -M_EPSILON && det < M_EPSILON) return false
            if det > -M_EPSILON and det < M_EPSILON:
                #return np.inf
                continue

            # invDet = 1.0f / det
            invDet = 1.0 / det

            # T = ray.origin - v0
            T_x = ray_origin_x - v0[0]
            T_y = ray_origin_y - v0[1]
            T_z = ray_origin_z - v0[2]

            # u = dot(T, P)*invDet
            u = invDet * (T_x*P_x + T_y*P_y + T_z*P_z)

            # if (u < 0.0f || u > 1.0f) return false
            if u < 0.0 or u > 1.0:
                #return np.inf
                continue

            # Q = cross(T, e1)
            Q_x = T_y*e1_z - T_z*e1_y
            Q_y = T_z*e1_x - T_x*e1_z
            Q_z = T_x*e1_y - T_y*e1_x

            # v = dot(ray.direction, Q)*invDet
            v = invDet * (ray_dir_x*Q_x + ray_dir_y*Q_y + ray_dir_z*Q_z)

            # if (v < 0.0f || u + v > 1.0f) return false
            if v < 0.0 or u+v > 1.0:
                #return np.inf
                continue

            # t0 = dot(e2, Q) * invDet
            t0 = invDet * (e2_x*Q_x + e2_y*Q_y + e2_z*Q_z)

            #if (t0 > M_EPSILON && t0 < t) {
            #    t = t0;
            #    intersection.position = ray.origin + t*ray.direction;
            #    intersection.normal = normalize(cross(e1, e2));
            #    return true
            #}
            
            if t0 > M_EPSILON and t0 < nearest_dist:
                #return t0
                nearest_dist = t0
            
            
        else:
            #if node[AABB_XMIN] <= ray_origin_x <= node[AABB_XMAX] and node[AABB_YMIN] <= ray_origin_y <= node[AABB_YMAX]:
            #    nearest_dist = 0.0
           
            # directionInv = glm::vec3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z)

            '''
            if ray_dir_x == 0.0:
                inv_x = np.inf
            else:
                inv_x = 1.0 / ray_dir_x
            
            if ray_dir_y == 0.0:
                inv_y = np.inf
            else:
                inv_y = 1.0 / ray_dir_y

            if ray_dir_z == 0.0:
                inv_z = np.inf
            else:
                inv_z = 1.0 / ray_dir_z
            '''

            #float t1 = (pmin.x - ray.origin.x) * inv.x;
            
            t1 = (node[AABB_XMIN] - ray_origin_x)
            t2 = (node[AABB_XMAX] - ray_origin_x)

            t3 = (node[AABB_YMIN] - ray_origin_y)
            t4 = (node[AABB_YMAX] - ray_origin_y)

            t5 = (node[AABB_ZMIN] - ray_origin_z)
            t6 = (node[AABB_ZMAX] - ray_origin_z)

            t1 = fix_t_gpu(t1, ray_dir_x)
            t2 = fix_t_gpu(t2, ray_dir_x)

            t3 = fix_t_gpu(t3, ray_dir_y)
            t4 = fix_t_gpu(t4, ray_dir_y)

            t5 = fix_t_gpu(t5, ray_dir_z)
            t6 = fix_t_gpu(t6, ray_dir_z)
            

            '''
            if ray_dir_x != 0.0:
                t1 = (node[AABB_XMIN] - ray_origin_x) * inv_x
                t2 = (node[AABB_XMAX] - ray_origin_x) * inv_x
            else:
                t1 = np.inf
                t2 = np.inf

            if ray_dir_y != 0.0:
                t3 = (node[AABB_YMIN] - ray_origin_y) * inv_y
                t4 = (node[AABB_YMAX] - ray_origin_y) * inv_y
            else:
                t3 = np.inf
                t4 = np.inf

            if ray_dir_z != 0.0:
                t5 = (node[AABB_ZMIN] - ray_origin_z) * inv_z
                t6 = (node[AABB_ZMAX] - ray_origin_z) * inv_z
            else:
                t5 = np.inf
                t6 = np.inf
            '''
            
            
            #float tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
            tmin = max(max(min(t1,t2), min(t3,t4)), min(t5,t6))
            #float tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));
            tmax = min(min(max(t1,t2), max(t3,t4)), max(t5,t6))

            if tmax <= 0.0 or tmin > tmax:
                #return np.inf
                continue
            

            #if not (node[AABB_XMIN] <= ray_origin_x <= node[AABB_XMAX] and node[AABB_YMIN] <= ray_origin_y <= node[AABB_YMAX]):
            #    continue

            #nearest_dist = 0.0

            #left_dist = get_nearest(aabbtree, int(node[AABB_LEFT]), ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, triangles)
            #right_dist = get_nearest(aabbtree, int(node[AABB_RIGHT]), ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, triangles)

            #return min(left_dist, right_dist)

            
            stack[xpix,ypix,stack_pointer] = int(node[AABB_LEFT])
            #stack[stack_pointer] = node_data[NODE_LEFT]
            #print(int(node[AABB_LEFT]))
            stack_pointer += 1

            stack[xpix,ypix,stack_pointer] = int(node[AABB_RIGHT])
            #stack[stack_pointer] = node_data[NODE_RIGHT]
            stack_pointer += 1
            
            #stack_pointer -= 1

    return nearest_dist



@cuda.jit('void(float64, float64, float64, float64, float64, int64, float64[:,:,:], int64[:,:,:], float64[:,:], int64, float64[:,:])')
def trace_aabb(x_origin, y_origin, x_range, y_range, zval, n_rays, triangles, stack, aabbtree, num_rays, image):

    xpix, ypix = cuda.grid(2)

    ray_origin_x = x_origin + (xpix * x_range / image.shape[0])
    ray_origin_y = y_origin + (ypix * y_range / image.shape[1])
    ray_origin_z = zval

    ray_dir_x = 0.0
    ray_dir_y = 0.0
    ray_dir_z = 1.0

    #nearest = get_nearest_save2(aabbtree, xpix, ypix, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, stack, triangles)
    nearest = np.inf

    n = 0

    for r in range(num_rays):
        ray_dist = get_nearest(aabbtree, xpix, ypix, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, stack, triangles)
        if ray_dist < nearest:
            nearest = ray_dist
        n += 1


    if nearest < np.inf:
        image[xpix, ypix] = n#1.0


'''
def inorderTraversal(root, parent_idx, lr, lst):

    self_index = len(lst)
    self_entry = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

    self_entry[AABB_PARENT] = parent_idx
    self_entry[AABB_XMIN] = root.aabb.limits[0][0]
    self_entry[AABB_XMAX] = root.aabb.limits[0][1]
    self_entry[AABB_YMIN] = root.aabb.limits[1][0]
    self_entry[AABB_YMAX] = root.aabb.limits[1][1]
    self_entry[AABB_ZMIN] = root.aabb.limits[2][0]
    self_entry[AABB_ZMAX] = root.aabb.limits[2][1]

    lst.append(self_entry)

    if parent_idx > -1:
        lst[parent_idx][lr] = self_index

    if (root.value and not root.is_leaf) or (root.is_leaf and not root.value):
        #print(root.value)
        print(root.aabb)

        print('this happens')
        exit()

    if root.is_leaf:
        lst[self_index][AABB_DATA] = root.value
        #print(root.value)
        return

    inorderTraversal(root.left, self_index, AABB_LEFT, lst)
    inorderTraversal(root.right, self_index, AABB_RIGHT, lst)
'''

def serialize_tree(root, parent_idx, lr, lst):
    self_index = len(lst)
    self_entry = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

    self_entry[AABB_PARENT] = parent_idx
    self_entry[AABB_XMIN] = root.bbox.xmin
    self_entry[AABB_XMAX] = root.bbox.xmax
    self_entry[AABB_YMIN] = root.bbox.ymin
    self_entry[AABB_YMAX] = root.bbox.ymax
    self_entry[AABB_ZMIN] = root.bbox.zmin
    self_entry[AABB_ZMAX] = root.bbox.zmax

    volume = (root.bbox.xmax - root.bbox.xmin) * (root.bbox.ymax - root.bbox.ymin) * (root.bbox.zmax - root.bbox.zmin)

    #print(self_entry)
    #print(volume)

    lst.append(self_entry)

    if parent_idx > -1:
        lst[parent_idx][lr] = self_index

    if (root.value is not None and not root.is_leaf()) or (root.is_leaf() and root.value is None):
        #print(root.value)
        #print(root.aabb)

        print('this happens')
        print(root.value is not None and not root.is_leaf())
        print(root.is_leaf() and not root.value is None)
        exit()

    if root.is_leaf():
        lst[self_index][AABB_DATA] = root.value
        #print(root.value)
        return

    serialize_tree(root.left, self_index, AABB_LEFT, lst)
    serialize_tree(root.right, self_index, AABB_RIGHT, lst)



class Bbox:
    def __init__(self, xmin, ymin, zmin, xmax, ymax, zmax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

class TreeNode:
    def __init__(self, bbox, left=None, right=None, value=None):
        self.bbox = bbox
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

    #def set_left(self, node):
    #    self.left = node
    
    #def set_right(self, node):
    #    self.right = node

    #def set_value(self, value):
    #    self.value = value

# 0: centroid x
# 1: centroid y
# 2: centroid z
# 3: bbox xmin
# 4: bbox ymin
# 5: bbox zmin
# 6: bbox xmax
# 7: bbox ymax
# 8: bbox zmax
# 9: index in x-sorted list
# 10: index in y-sorted list
# 11: index in z-sorted list
# 12: index in triangle list
def build_tree(tris, indexes, centroid_x_order, centroid_y_order, centroid_z_order, x_points_list, y_points_list):
    #print('index length')
    #print(len(indexes))

    if len(indexes) == 0:
        node = None
        #return None
    elif len(indexes) == 1:
        tri = tris[indexes[0]]
        node_bbox = Bbox(tri[3],tri[4],tri[5],tri[6],tri[7],tri[8])
        #volume = (node_bbox.xmax - node_bbox.xmin) * (node_bbox.ymax - node_bbox.ymin) * (node_bbox.zmax - node_bbox.zmin)
        #if volume == 0.0:
            #print('???')
            #print(tri)
            #print(indexes)
            #exit()
        node = TreeNode(node_bbox, value=indexes[0])#value=tri[12])
        #print('node value')
        #print(node.value)
        #return node

    else:
        index_tris = tris[indexes]
        index_centroid_x = centroid_x_order[ (index_tris[:,9]).astype('int64') ]
        index_centroid_y = centroid_y_order[ (index_tris[:,10]).astype('int64') ]
        index_centroid_z = centroid_z_order[ (index_tris[:,11]).astype('int64') ]

        tris_xmin = np.min(index_tris[:,3])
        tris_xmax = np.max(index_tris[:,6])

        tris_ymin = np.min(index_tris[:,4])
        tris_ymax = np.max(index_tris[:,7])

        tris_zmin = np.min(index_tris[:,5])
        tris_zmax = np.max(index_tris[:,8])

        node_bbox = Bbox(tris_xmin, tris_ymin, tris_zmin, tris_xmax, tris_ymax, tris_zmax)

        x_points_list += [tris_xmin, tris_xmax]
        y_points_list += [tris_ymin, tris_ymax]

        xrange = tris_xmax - tris_xmin
        yrange = tris_ymax - tris_ymin
        zrange = tris_zmax - tris_zmin

        if xrange > yrange and xrange > zrange:
            axis = 0
            order_list = index_centroid_x
        elif yrange > xrange and yrange > zrange:
            axis = 1
            order_list = index_centroid_y
        else:
            axis = 2
            order_list = index_centroid_z

        # first_half_indexes = (centroid_x_order[:n_tris // 2, 0]).astype('int64')
        # first_half_x = tri_structs[first_half_indexes]
        # last_half_indexes = (centroid_x_order[n_tris // 2:, 0]).astype('int64')
        # last_half_x = tri_structs[last_half_indexes]
        #print(order_list)

        #print('indexes')
        #print(indexes)
        #print('order list')
        #print(order_list)
        #print('midpt')
        #print(len(order_list)//2)
        #print('left tris?')
        #print(order_list[:len(order_list)//2, 0])
        #print('left tris')
        left_tris = (order_list[:len(order_list)//2, 0]).astype('int64')
        #print('right tris')
        right_tris = (order_list[len(order_list)//2:, 0]).astype('int64')
        #print('...')

        node = TreeNode(node_bbox)

        '''
        print('asfsadfsdf')
        print(left_tris)
        print(right_tris)
        print('xzcvxcv')
        '''

        if len(left_tris) > 0:
            node.left = build_tree(tris, left_tris, centroid_x_order, centroid_y_order, centroid_z_order, x_points_list, y_points_list)
            #print('node.left')
            #print(node.left)
        if len(right_tris) > 0:
            node.right = build_tree(tris, right_tris, centroid_x_order, centroid_y_order, centroid_z_order, x_points_list, y_points_list)
            #print('node.right')
            #print(node.right)

        #return node
    
    if (node.value is not None and not node.is_leaf()) or (node.is_leaf() and node.value is None):
        print('***')
        print(indexes)
        print(len(indexes))
        print(node.value is not None and not node.is_leaf())
        print(node.is_leaf() and node.value is None)
        exit()
    

    return node

        

        


img_dim = 512
threads = 16
tpb = (threads, threads)
bpg = ( int(img_dim / threads), int(img_dim / threads) )
image = cuda.to_device(np.zeros((img_dim, img_dim), dtype='f8'))

x_orig = 3.5e10
x_range = 8e10 - x_orig
y_orig = 3.5e10
y_range = 8e10 - y_orig

z_orig = 3.4e+10
z_range = 8e10 - z_orig


obj_dir = os.path.join(os.path.dirname(__file__), 'data/obj_files/frame_498')

start = time.time()


for fname in ['big']:
    mesh = meshio.read(os.path.join(obj_dir, f'{fname}.obj'))

    #tree = AABBTree()

    tris = np.zeros((len(mesh.cells[0].data), 3, 3), dtype='f8')

    # 0: centroid x
    # 1: centroid y
    # 2: centroid z
    # 3: bbox xmin
    # 4: bbox ymin
    # 5: bbox zmin
    # 6: bbox xmax
    # 7: bbox ymax
    # 8: bbox zmax
    # 9: index in x-sorted list
    # 10: index in y-sorted list
    # 11: index in z-sorted list
    # 12: index in triangle list
    tri_structs = np.zeros((len(tris), 13), dtype='f8')

    for tri_idx, tri in enumerate(mesh.cells[0].data):
        tris[tri_idx,0] = mesh.points[tri[0]]
        tris[tri_idx,1] = mesh.points[tri[1]]
        tris[tri_idx,2] = mesh.points[tri[2]]

        xmin = np.min(tris[tri_idx,:,0])
        xmax = np.max(tris[tri_idx,:,0])

        ymin = np.min(tris[tri_idx,:,1])
        ymax = np.max(tris[tri_idx,:,1])

        zmin = np.min(tris[tri_idx,:,2])
        zmax = np.max(tris[tri_idx,:,2])

        #tree.add(AABB( [(xmin,xmax), (ymin,ymax), (zmin,zmax)] ), tri_idx)

        x_centroid = (1.0/3) * (tris[tri_idx,0,0] + tris[tri_idx,1,0] + tris[tri_idx,2,0])
        y_centroid = (1.0/3) * (tris[tri_idx,0,1] + tris[tri_idx,1,1] + tris[tri_idx,2,1])
        z_centroid = (1.0/3) * (tris[tri_idx,0,2] + tris[tri_idx,1,2] + tris[tri_idx,2,2])

        tri_structs[tri_idx,0] = x_centroid
        tri_structs[tri_idx,1] = y_centroid
        tri_structs[tri_idx,2] = z_centroid

        tri_structs[tri_idx,3] = xmin
        tri_structs[tri_idx,4] = ymin
        tri_structs[tri_idx,5] = zmin

        tri_structs[tri_idx,6] = xmax
        tri_structs[tri_idx,7] = ymax
        tri_structs[tri_idx,8] = zmax

        tri_structs[tri_idx,12] = tri_idx

    n_tris = len(tri_structs)

    started = time.time()
    centroid_x_order = np.array(sorted([(idx, elem[0]) for idx, elem in enumerate(tri_structs)], key=lambda x: x[1]))
    centroid_y_order = np.array(sorted([(idx, elem[1]) for idx, elem in enumerate(tri_structs)], key=lambda x: x[1]))
    centroid_z_order = np.array(sorted([(idx, elem[2]) for idx, elem in enumerate(tri_structs)], key=lambda x: x[1]))

    for centroid_order_idx in range(len(centroid_x_order)):
        tri_idx = int(centroid_x_order[centroid_order_idx,0])
        tri_structs[tri_idx,9] = centroid_order_idx
        
    for centroid_order_idx in range(len(centroid_y_order)):
        tri_idx = int(centroid_y_order[centroid_order_idx,0])
        tri_structs[tri_idx,10] = centroid_order_idx

    for centroid_order_idx in range(len(centroid_z_order)):
        tri_idx = int(centroid_z_order[centroid_order_idx,0])
        tri_structs[tri_idx,11] = centroid_order_idx

    #print(tri_structs[centroid])
    print(centroid_x_order[:100])
    first_half_indexes = (centroid_x_order[:n_tris // 2, 0]).astype('int64')
    first_half_x = tri_structs[first_half_indexes]
    last_half_indexes = (centroid_x_order[n_tris // 2:, 0]).astype('int64')
    last_half_x = tri_structs[last_half_indexes]

    print(np.max(first_half_x[:,0]) < np.min(last_half_x[:,0]))

    x_points_list = []
    y_points_list = []

    tree = build_tree(tri_structs, (tri_structs[:,12]).astype('int64'), centroid_x_order, centroid_y_order, centroid_z_order, x_points_list, y_points_list)

    
    plt.scatter(x_points_list, y_points_list)
    plt.show()
    
    
    print(time.time() - started)

    tree_lst = []
    serialize_tree(tree, -1, -1, tree_lst)
    print(len(tree_lst))
    print(len(tris))

    tree_serial = np.array(tree_lst)
    #exit()

    

    d_stack = cuda.to_device(np.zeros((img_dim, img_dim, 100)).astype('int64'))
    d_tree = cuda.to_device(tree_serial.astype('f8'))

    n_rays = 1

    d_tris = cuda.to_device(tris.astype('f8'))

    tree_ints = np.zeros((len(tree_serial), 3)).astype('int64')
    tree_ints[:,NODE_LEFT] = tree_serial[:,AABB_LEFT].astype('int64')
    tree_ints[:,NODE_RIGHT] = tree_serial[:,AABB_RIGHT].astype('int64')

    d_tree_ints = cuda.to_device(tree_ints)

    gpu_start = time.time()
    num_rays = 10000

    #trace[bpg, tpb](x_orig, y_orig, x_range, y_range, z_orig, d_tris, 1, image)
    trace_aabb[bpg, tpb](x_orig, y_orig, x_range, y_range, z_orig, n_rays, d_tris, d_stack, d_tree, num_rays, image)

    print('gpu time')
    print(time.time() - gpu_start)

    img = image.copy_to_host()
    plt.imshow(img)
    plt.show()

    #exit()

    '''
    img_cpu = np.zeros((img_dim,img_dim))
    for xpix in range(img_dim):
        print(xpix)
        for ypix in range(img_dim):

            ray_origin_x = x_orig + (xpix * x_range / image.shape[0])
            ray_origin_y = y_orig + (ypix * y_range / image.shape[1])
            ray_origin_z = z_orig

            ray_dir_x = 0.0
            ray_dir_y = 0.0
            ray_dir_z = 1.0

            complete_list = np.zeros((len(tree_serial),))

            #min_dist = get_nearest(tree_serial, 0, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, tris)
            stack = np.zeros((len(tris),)).astype('int64')
            #min_dist = get_nearest(tree_serial, complete_list, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, tris)
            min_dist = get_nearest(tree_serial, ray_origin_x, ray_origin_y, ray_origin_z, ray_dir_x, ray_dir_y, ray_dir_z, stack, tris)

            if min_dist < np.inf:
                img_cpu[xpix, ypix] = 1.0


    plt.imshow(img_cpu)
    plt.show()
    '''
    

print(time.time() - start)



'''
start = time.time()

grid_dim = 20

#for fname in ('big', 'small', 'boundary'):
for fname in ['small']:
    mesh = meshio.read(os.path.join(obj_dir, f'{fname}.obj'))

    tree = AABBTree()

    grid_idx = np.zeros((grid_dim, grid_dim, grid_dim), dtype='int64')
    grid = np.zeros((grid_dim, grid_dim, grid_dim, len(mesh.cells[0].data) // 100), dtype='int64') - 1

    grid_x_orig = np.min(mesh.points[:,0])
    grid_x_range = np.max(mesh.points[:,0]) + 0.1 - grid_x_orig
    grid_y_orig = np.min(mesh.points[:,1])
    grid_y_range = np.max(mesh.points[:,1]) + 0.1 - grid_y_orig
    grid_z_orig = np.min(mesh.points[:,2])
    grid_z_range = np.max(mesh.points[:,2]) + 0.1 - grid_z_orig

    print(grid.shape)
    print(grid_x_orig)

    tri_grid_bounds = np.zeros((3,2))
    tri_grid_bounds[0,0] = grid_x_orig
    tri_grid_bounds[0,1] = grid_x_orig + grid_x_range
    tri_grid_bounds[1,0] = grid_y_orig
    tri_grid_bounds[1,1] = grid_y_orig + grid_y_range
    tri_grid_bounds[2,0] = grid_z_orig
    tri_grid_bounds[2,1] = grid_z_orig + grid_z_range

    print(tri_grid_bounds)

    
    tris = np.zeros((len(mesh.cells[0].data), 3, 3), dtype='f8')
    for tri_idx, tri in enumerate(mesh.cells[0].data):
        tris[tri_idx,0] = mesh.points[tri[0]]
        tris[tri_idx,1] = mesh.points[tri[1]]
        tris[tri_idx,2] = mesh.points[tri[2]]

        xmin = np.min(tris[tri_idx,:,0])
        xmax = np.min(tris[tri_idx,:,0])

        ymin = np.min(tris[tri_idx,:,1])
        ymax = np.min(tris[tri_idx,:,1])

        zmin = np.min(tris[tri_idx,:,2])
        zmax = np.min(tris[tri_idx,:,2])

        xmin_voxel = int(np.floor( ( (xmin - grid_x_orig) / grid_x_range ) * grid_dim ))
        xmax_voxel = int(np.floor( ( (xmax - grid_x_orig) / grid_x_range ) * grid_dim ))

        ymin_voxel = int(np.floor( ( (ymin - grid_y_orig) / grid_y_range ) * grid_dim ))
        ymax_voxel = int(np.floor( ( (ymax - grid_y_orig) / grid_y_range ) * grid_dim ))

        zmin_voxel = int(np.floor( ( (zmin - grid_z_orig) / grid_z_range ) * grid_dim ))
        zmax_voxel = int(np.floor( ( (zmax - grid_z_orig) / grid_z_range ) * grid_dim ))

        #grid[xmin_voxel,ymin_voxel,zmin_voxel,grid_idx[xmin_voxel,ymin_voxel,zmin_voxel]] = tri_idx
        #grid_idx[xmin_voxel,ymin_voxel,zmin_voxel] += 1

        for x_voxel in range(xmin_voxel, xmax_voxel+1):
            for y_voxel in range(ymin_voxel, ymax_voxel+1):
                for z_voxel in range(zmin_voxel, zmax_voxel+1):
                    #print(x_voxel)
                    #print(y_voxel)
                    #print(z_voxel)
                    #print(grid_idx[x_voxel,y_voxel,z_voxel])

                    grid[x_voxel,y_voxel,z_voxel,grid_idx[x_voxel,y_voxel,z_voxel]] = tri_idx
                    grid_idx[x_voxel,y_voxel,z_voxel] += 1

        

        #tree.add(AABB(tris[tri_idx]), f'box {tri_idx}')
        #tree.add(AABB( [(xmin,xmax), (ymin,ymax), (zmin,zmax)] ), tri_idx)

    print(tris)



    d_tris = cuda.to_device(tris.astype('f8'))
    d_tri_grid = cuda.to_device(grid)
    d_grid_bounds = cuda.to_device(tri_grid_bounds.astype('f8'))
    nrays = 100

    for grid_z in range(grid_dim):
        grid_z_min = tri_grid_bounds[2,0] + (grid_z / grid_dim) * (tri_grid_bounds[2,1] - tri_grid_bounds[2,0])
        grid_z_max = tri_grid_bounds[2,0] + ( (grid_z+1) / grid_dim) * (tri_grid_bounds[2,1] - tri_grid_bounds[2,0])

        print((grid_z_min, grid_z_max))
    print(tri_grid_bounds)

    trace[bpg, tpb](x_orig, y_orig, x_range, y_range, z_orig, d_tris, d_tri_grid, d_grid_bounds, nrays, image)

img = image.copy_to_host()
plt.imshow(img)
print(time.time() - start)
plt.show()
'''