import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
#from scipy.spatial.
from shapely.geometry import Polygon, Point
from shapely import errors as se
from shapely.plotting import plot_polygon
import scipy
import random

def segment_intersect(p1, p2, p3, p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)

points_border = []

n_angles = 100

np.random.seed(18)
#np.random.seed(13)

necessary_pt_0 = [np.sqrt(2) * np.sin(0)**3, - np.cos(0)**3 - np.cos(0)**2 + 2*np.cos(0)]
necessary_pt_1 = [np.sqrt(2) * np.sin(np.pi)**3, - np.cos(np.pi)**3 - np.cos(np.pi)**2 + 2*np.cos(np.pi)]
#print(necessary_pt_0)
#print(necessary_pt_1)
#exit()


for i in range(n_angles):
    t = i * (2*np.pi / n_angles)
    #t = np.pi*0.5 + (np.pi / n_angles)

    x = np.sqrt(2) * np.sin(t)**3
    y = - np.cos(t)**3 - np.cos(t)**2 + 2*np.cos(t)

    #x_vals.append(x)
    #y_vals.append(y)

    points_border.append([x,y])


'''
for i in range(30):
    t = i * (2 * np.pi / 30)

    x = np.cos(t)
    y = np.sin(t)

    points_border.append([x,y])
'''


border_edges = [ [points_border[i],points_border[i+1]] for i in range(len(points_border) - 1)]
border_edges.append([points_border[-1], points_border[0]])

print(border_edges)
#exit()


points_center = []

x_min = -1.5
x_max = 1.5
y_min = -2.1
y_max = 0.7

boundary_polygon = Polygon(points_border )


while len(points_center) < 65:
    #x = -1 + 2*np.random.random()
    #y = -1 + 2*np.random.random()
    x = x_min + np.random.random()*(x_max-x_min)
    y = y_min + np.random.random()*(y_max-y_min)

    #if np.sqrt(x**2 + y**2) < 0.9:
    #    points_center.append([x,y])
    if boundary_polygon.contains(Point(x, y)):
        points_center.append([x,y])


'''
while len(points_center) < 65:
    x = -1 + 2*np.random.random()
    y = -1 + 2*np.random.random()

    if np.sqrt(x**2 + y**2) < 1.0:
        points_center.append([x,y])
'''


all_points = np.array( points_center )
#all_points = np.array(points_center)

'''
tri = Delaunay(all_points)
plt.triplot(all_points[:,0], all_points[:,1], tri.simplices)

plt.scatter(all_points[:,0],all_points[:,1])
plt.show()
'''

#boundary_polygon_test = Polygon(random.shuffle(points_border))

#print(boundary_polygon_test)
print(boundary_polygon)
#exit()



for lloyd_iter in range(14):
    vor = Voronoi(all_points)
    #fig = voronoi_plot_2d(vor)
    #plt.show()

    region_area_vals = []

    new_points = []

    new_boundary_points = []

    for point_idx, point in enumerate(all_points):

        region_points_list = []

        #all_seed_neighbors = []
        point_ridge_indexes = []
        point_neighbor_indexes = []

        for ridge_idx, ridge_pts in enumerate(vor.ridge_points):
            if point_idx in ridge_pts:
                point_ridge_indexes.append(ridge_idx)
                neighbor_idx = ridge_pts[0] if ridge_pts[1] == point_idx else ridge_pts[1]
                point_neighbor_indexes.append(neighbor_idx)

        for i in range(len(point_neighbor_indexes)):
            
            neighbor_idx = point_neighbor_indexes[i]
            neighbor_pt = all_points[neighbor_idx]

            ridge_idx = point_ridge_indexes[i]

            rvs = vor.ridge_vertices[ridge_idx]

            if -1 in rvs:
                vor_vertex_idx = rvs[0] if rvs[1] == -1 else rvs[1]

                vor_vertex = vor.vertices[vor_vertex_idx]

                if not boundary_polygon.contains(Point(vor_vertex[0], vor_vertex[1])):
                    continue

                midpt = [0.5 * (point[0]+neighbor_pt[0]), 0.5 * (point[1]+neighbor_pt[1])]

                vec_x = midpt[0] - vor_vertex[0]
                vec_y = midpt[1] - vor_vertex[1]
                vec_norm = np.sqrt(vec_x**2 + vec_y**2)
                vec_x /= vec_norm
                vec_y /= vec_norm

                new_x = vor_vertex[0] + 10 * vec_x
                new_y = vor_vertex[1] + 10 * vec_y

                all_boundary_intersections = []

                for edge_idx, edge in enumerate(border_edges):
                    intersection = segment_intersect(edge[0], edge[1], vor_vertex, [new_x, new_y])
                    if intersection is not None:
                        all_boundary_intersections.append(intersection)

                intersection_dists = np.array([ np.sqrt( (elem[0]-vor_vertex[0])**2 + (elem[1]-vor_vertex[1])**2 ) for elem in all_boundary_intersections  ])

                closest_intersection = all_boundary_intersections[np.argmin(intersection_dists)]
                region_points_list.append(list(closest_intersection))
                new_boundary_points.append(list(closest_intersection))

            else:
                if vor.vertices[rvs[0]].tolist() not in region_points_list:
                    region_points_list.append(vor.vertices[rvs[0]].tolist())
                if vor.vertices[rvs[1]].tolist() not in region_points_list:
                    region_points_list.append(vor.vertices[rvs[1]].tolist())

        ordered_region_points = []

        region_point_thetas = [(i, np.arctan2(elem[1] - point[1], elem[0] - point[0])) for i, elem in enumerate(region_points_list)]
        region_point_thetas = sorted(region_point_thetas, key=lambda x: x[1])

        for i, _ in region_point_thetas:
            ordered_region_points.append(region_points_list[i])

        region_polygon = Polygon(ordered_region_points)


        try:
            x = boundary_polygon.intersection(region_polygon)
        except (se.TopologicalError, se.GEOSException, se.ShapelyError, se.DimensionError, se.GeometryTypeError) as error:
            print(error)
            exit()

        region_area_vals.append(abs(x.area - region_polygon.area))


        centroid = x.centroid

        try:
            centroid_x = centroid.x
        except:
            print(x)
            fig = plot_polygon(x)
            plt.show()


        new_points.append([centroid.x, centroid.y])

    all_points = np.array(new_points)

all_points = all_points.tolist()
#all_points.append(necessary_pt_0)
#all_points.append(necessary_pt_1)

all_points_final = np.array(all_points+[[0,-0.06]])#np.array(all_points + [[-1.173,-0.259], [-1.267,0.258],[-1.06,0.501],[-0.815,0.532]]   )#np.array( all_points.tolist() + new_boundary_points)

boundary_points = np.array(points_border)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(boundary_points[:,0], boundary_points[:,1])
ax1.scatter(all_points_final[:,0], all_points_final[:,1])
plt.show()


vor = Voronoi(all_points_final)
fig = voronoi_plot_2d(vor)
plt.show()

tri = Delaunay(all_points_final)

remove_indexes = []
face_list = []
for face_idx, face in enumerate(tri.simplices):
    p0 = all_points_final[face[0]]
    p1 = all_points_final[face[1]]
    p2 = all_points_final[face[2]]

    if p0[0]*p1[0] < 0 and p0[1] > 0.03:
        remove_indexes.append(face_idx)
    elif p0[0]*p2[0] < 0 and p0[1] > 0.03:
        remove_indexes.append(face_idx)
    elif p2[0]*p1[0] < 0 and p2[1] > 0.03:
        remove_indexes.append(face_idx)
    else:
        face_list.append(face)

print(len(remove_indexes))


plt.triplot(all_points_final[:,0], all_points_final[:,1], face_list, color='black')
plt.show()


print(region_area_vals)


            

