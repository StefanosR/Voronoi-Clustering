import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import scipy
import scipy.spatial
import scipy.optimize 

arr = [(52, 12, 13), (13, 12, -2), (-21, 8, 2), (12, 10, 7), (13, 22, 6), (7, 11, -16), (-10, -5, 0), (-14, -3, 4), (20, 11, 8), (12, 4, 12)]

# air_traffic_mess = np.random.random_sample((10,3))

air_traffic_mess = np.array(arr)

#the edges of a 3D Voronoi diagram will be the farthest from the obstacle coordinates
vor = scipy.spatial.Voronoi(air_traffic_mess)
        
fig_drone_3d = plt.figure()
fig_drone_3d.set_size_inches(8,8)
ax = fig_drone_3d.add_subplot(111, projection = '3d')

for ridge_indices in vor.ridge_vertices:
    voronoi_ridge_coords = vor.vertices[ridge_indices]
    ax.plot(voronoi_ridge_coords[...,0], voronoi_ridge_coords[...,1], voronoi_ridge_coords[...,2], lw=2, c = 'blue', alpha=0.2)
    
vor_vertex_coords = vor.vertices

# obstacles
ax.scatter(air_traffic_mess[...,0], air_traffic_mess[...,1], air_traffic_mess[...,2], c= 'k', label='obstacles', edgecolor='none')

ax.scatter(vor_vertex_coords[...,0], vor_vertex_coords[...,1], vor_vertex_coords[...,2], c= 'orange', label='Voronoi vertices',edgecolors='white', marker = 'o', alpha = 0.9)

ax.legend()

# infinity
ax.set_xlim3d(air_traffic_mess[...,0].min(), air_traffic_mess[...,0].max())
ax.set_ylim3d(air_traffic_mess[...,1].min(), air_traffic_mess[...,1].max())
ax.set_zlim3d(air_traffic_mess[...,2].min(), air_traffic_mess[...,2].max())

plt.show()