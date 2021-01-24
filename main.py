import numpy as np
import math
import pylab as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.animation as animation
from matplotlib import collections as mc
from shapely.geometry import  LineString



# Info 
# Algorithm: Delaunay Triangulation
# List triangulation
# Add triangle that envelops all the points in the triangulation list
# 1. Add point
# 2. Find all triangles where the new point is in (bad triangles)
# 3. Find all edges that are between bad triangle and good triangle
# 4. Construct new triangles with the edges
# 5. Remove bad triangles
# 6. Repeat until no more points can be added
# 7. Remove all triangles that have a vertex from the super triangle

# For transforming Delauney -> Voronoi: (UPDATE HERE)
# 1. Find all the circumcenters of the tringles
# 2. Connect adjacent triangle circumcenters with edge.

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Imported data will be assigned here as points
class Point:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

# Consists of the points and triangles of each edge (2)
class Edge:
    def __init__(self, points=[], trigs=[]):
        self.points = points
        self.trigs = trigs
    # Draw the edge
    def toArtist(self):
        points = np.array(list(map(lambda p: np.asarray([p.x, p.y]), self.points())))
        return plt.plot(points[:2, :], color='C0', alpha=0.8, fill=False, clip_on=True, linewidth=1)

# Describes the Triangle class
class Triangle:
    # Initialize a Triangle
    def __init__(self, edges=[]):
        self.edges = edges
        for e in edges:
            e.trigs.append(self)
    # Triangle consists of 3 corners
    def points(self):
        return [self.edges[0].points[0], self.edges[1].points[0], self.edges[2].points[0]]
    # Draw the triangle function
    def toArtist(self):
        points = np.array(list(map(lambda p: np.asarray([p.x, p.y]), self.points())))
        return plt.Polygon(points[:3, :], color='C0', alpha=0.8, fill=False, clip_on=True, linewidth=1)

#Describes the Tetrahedron class
class Tetrahedron:
    def __init__(self, triangles=[]):
        self.triangles = triangles

    def frompoints(self, pnts=[]):
        points = pnts
        p1 = points[0]
        p2 = points[1]
        p3 = points[2]
        p4 = points[3]
        
        e01 = Edge([p1, p2]) # edge 1 of trig 0
        e02 = Edge([p2, p3])
        e03 = Edge([p3, p1])
        e11 = Edge([p3, p2])
        e12 = Edge([p2, p4])
        e13 = Edge([p4, p3])
        e21 = Edge([p3, p4])
        e22 = Edge([p4, p1])
        e23 = Edge([p1, p3])
        e31 = Edge([p1, p4])
        e32 = Edge([p4, p2])
        e33 = Edge([p2, p1])

        t0 = Triangle([e01, e02, e03]) #aristera
        t1 = Triangle([e11, e12, e13]) #kato
        t2 = Triangle([e21, e22, e23]) #piso
        t3 = Triangle([e31, e32, e33]) #brosta
        self.triangles =[t0,t1,t2,t3]
        return True


    def edges(self):
        edges=[]
        for t in self.triangles:
            for e in t.edges:
                flag = False
                if edges:
                    for e2 in edges:
                        if e.points[0].x == e2.points[0].x and e.points[0].y == e2.points[0].y and e.points[1].x == e2.points[1].x and e.points[1].y == e2.points[1].y:  
                            flag = True
                        elif e.points[0].x == e2.points[1].x and e.points[0].y == e2.points[1].y and e.points[1].x == e2.points[0].x and e.points[1].y == e2.points[0].y:
                            flag = True
                    if not flag:
                        edges.append(e)
                else:
                    edges.append(e)
        # for ed in edges:
        #     print("printing points of edges() point 1:", ed.points[0].x,ed.points[0].y,ed.points[0].z, " point 2:",ed.points[1].x,ed.points[1].y, ed.points[1].z) 
        # print("end")              
        return edges
    
    def points(self):
        points=[]
        
        for e in self.edges():
            flag = False
            if points:
                for p in e.points:
                    for p2 in points:
                        if p==p2:
                            flag = True
                    if not flag:
                        points.append(p)
            else :
                points.append(e.points[0])
                points.append(e.points[1])
                # print("added first 2 points", points[0].x,  points[0].y,  points[0].z,  points[1].x,  points[1].y,  points[1].z )
        return points

    def painttet(self):
        points = np.array(list(map(lambda p: np.asarray([p.x, p.y,p.z]), self.points())))
        verts= [ [points[0], points[1], points[2]],
         [points[2], points[1], points[3]],
          [points[2], points[3], points[0]],
           [points[0], points[3], points[1]] ]
        axis.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
        return True

# Circles of the Delaunay triangulation
class Circle:
    # Initialize Circle
    def __init__(self, x=0, y=0, radius=0):
        self.x = x
        self.y = y
        self.radius = radius
    # Create Circle
    def fromTriangle(self, t):
        pnts = t.points()
        p1 = [pnts[0].x, pnts[0].y]
        p2 = [pnts[1].x, pnts[1].y]
        p3 = [pnts[2].x, pnts[2].y]
        # Math
        temp = p2[0] * p2[0] + p2[1] * p2[1]                # (x2^2 + y2^2)
        bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2     # (x1^2 + y1^2 - x2^2 + y2^2)/2
        cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2     # ((x2^2 + y2^2) - x3^2 - y3^2)/2

        # Det = (x1-x2)*(y2-y3) - (x2-x3)*(y1-y2)
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1]) # or (p1[0] - p2[0]) * (p2[1] - p3[1]) - \ (p2[0] - p3[0]) * (p1[1] - p2[1])
        
        if abs(det) < 1.0e-6: # if 3 points are aligned, there can't be triangle
            return False

        self.x = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
        self.y = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
        self.radius = np.sqrt((self.x - p1[0])**2 + (self.y - p1[1])**2)

        return True

    # Draw Circle
    def toArtist(self):
        return plt.Circle((self.x, self.y), self.radius, color='C0',
                          fill=False, clip_on=True, alpha=0.2)

class Sphere:
    # Initialize Sphere
    def __init__(self, x=0, y=0, z=0, radius=0):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
    # Create Sphere
    def fromTetrahedron(self, tet):
        pnts = tet.points()
        for p in points:
            print("Point:",p.x, p.y, p.z)
        
        p1 = [pnts[0].x, pnts[0].y, pnts[0].z]
        p2 = [pnts[1].x, pnts[1].y, pnts[1].z]
        p3 = [pnts[2].x, pnts[2].y, pnts[2].z]
        p4 = [pnts[3].x, pnts[3].y, pnts[3].z]
        print("points of tet:", p1, p2 , p3, p4)
        
        # Math
        t1 = - (p1[0]*p1[0] + p1[1]*p1[1] + p1[2]*p1[2])
        t2 = - (p2[0]*p2[0] + p2[1]*p2[1] + p2[2]*p2[2])
        t3 = - (p3[0]*p3[0] + p3[1]*p3[1] + p3[2]*p3[2])
        t4 = - (p4[0]*p4[0] + p4[1]*p4[1] + p4[2]*p4[2])

        print(" t1 = ",t1," t2 = ",t2," t3 = ",t3," t4 = ",t4,)

        #Define arrays for dets
        T = np.array([[p1[0], p1[1], p1[2], 1],
                     [p2[0], p2[1], p2[2], 1], 
                     [p3[0], p3[1], p3[2], 1],
                     [p4[0], p4[1], p4[2], 1]])
        
        detT = np.linalg.det(T)
        print("T = ", detT)
        
        D = np.array([[t1, p1[1], p1[2], 1],
                     [t2, p2[1], p2[2], 1], 
                     [t3, p3[1], p3[2], 1],
                     [t4, p4[1], p4[2], 1]])

        detD = np.linalg.det(D)
        print("D = ",detD)
        detD = detD / detT


        E = np.array([[p1[0], t1, p1[2], 1],
                     [p2[0], t2, p2[2], 1], 
                     [p3[0], t3, p3[2], 1],
                     [p4[0], t4, p4[2], 1]])
        
        detE = np.linalg.det(E)
        detE = detE / detT
        print("E = ",detE)

        F = np.array([[p1[0], p1[1], t1, 1],
                     [p2[0], p2[1], t2, 1], 
                     [p3[0], p3[1], t3, 1],
                     [p4[0], p4[1], t4, 1]])
        
        detF = np.linalg.det(F)
        detF = detF / detT

        G = np.array([[p1[0], p1[1], p1[2], t1],
                     [p2[0], p2[1], p2[2], t2], 
                     [p3[0], p3[1], p3[2], t3],
                     [p4[0], p4[1], p4[2], t4]])
        
        detG = np.linalg.det(G)
        print("G = ",detG)
        detG = detG / detT
        
        self.x = -detD / 2
        self.y = -detE / 2
        self.z = -detF / 2 
        self.radius = math.sqrt(detD*detD + detE*detE + detF*detF - 4*detG)/2
        
        return True
    
    def paintsphere(self):
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x=np.cos(u)*np.sin(v)
        y=np.sin(u)*np.sin(v)
        z=np.cos(v)
        # shift and scale sphere
        x = self.radius*x + self.x
        y = self.radius*y + self.y
        z = self.radius*z + self.z
        return (x,y,z)
        # axis.scatter(self.x, self.y, self.z, s=np.pi*(self.radius)**2*100, c='blue', alpha=0.75)
        # # plt.Sphere((self.x, self.y,self.z), self.radius, color='C0', fill=False, clip_on=True, alpha=0.2)
        # return True


# Function 1: Calculates the super triangle
def calculateSuperTetrahedron(points):
    # Find boundary box that contain all points
    p_min = Point(min(points, key=lambda p: p.x).x - 0.1,
                  min(points, key=lambda p: p.y).y - 0.1,
                  min(points, key=lambda p: p.z).z - 0.1,)
    p_max = Point(max(points, key=lambda p: p.x).x + 0.1,
                  max(points, key=lambda p: p.y).y + 0.1,
                  max(points, key=lambda p: p.z).z + 0.1)

    a = p_max.x - p_min.x  # "distance" between max and min points
    b = p_max.y - p_min.y
    c = p_max.z - p_min.z

    # p1 = Point(p_min.x - a, p_min.y - b, p_min.z - c)
    # p2 = Point(p_min.x, p_max.y + b, p_min.z)
    # p3 = Point(p_min.x, p_min.y, p_max.z + c)
    # p4 = Point(p_max.x + a, p_min.y, p_min.z)

    p1 = Point(0,0, 250)
    p2 = Point(-250,-250, -250)
    p3 = Point(250,-250, -250)
    p4 = Point(0,250, -250)

    points.insert(0, p1)
    points.insert(0, p2)
    points.insert(0, p3)
    points.insert(0, p4)

    
    t = Tetrahedron()
    t.frompoints([p1,p2,p3,p4])

    return t

# Function 2:
def pointInsideSphere(p, tet):
    
    s = Sphere()
    s.fromTetrahedron(tet)
    x1 = p.x
    y1 = p.y
    z1 = p.z
    x2 = s.x
    y2 = s.y
    z2 = s.z
    distance = math.sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1)) 

    # pnts = tet.points()

    # p3 = pnts[0]
    # p2 = pnts[1]
    # p1 = pnts[2]

    # x0 = p.x
    # y0 = p.y
    # x1 = p1.x
    # y1 = p1.y
    # x2 = p2.x
    # y2 = p2.y
    # x3 = p3.x
    # y3 = p3.y

    # ax_ = x1-x0
    # ay_ = y1-y0
    # bx_ = x2-x0
    # by_ = y2-y0
    # cx_ = x3-x0
    # cy_ = y3-y0

    # return (
    #     (ax_*ax_ + ay_*ay_) * (bx_*cy_-cx_*by_) -
    #     (bx_*bx_ + by_*by_) * (ax_*cy_-cx_*ay_) +
    #     (cx_*cx_ + cy_*cy_) * (ax_*by_-bx_*ay_)
    # ) > 0
    return distance < s.radius

# Function 3: Finds if a certain edge is shared with any triangle
def isSharedTrig(trig, tets, current_tet):
    shared_tets = []
    for tet in tets:
        if tet!=current_tet:
            flag = False
            for trig1 in tet.triangles:  # check if the vertices of the inserted edge are same with those of the triangle
                
                flag1 = False
                e1 = trig1.edges[0] 
                for e in trig.edges :
                    if e.points[0].x == e1.points[0].x and e.points[0].y == e1.points[0].y and e.points[1].x == e1.points[1].x and e.points[1].y == e1.points[1].y:  
                        flag1 = True
                    elif e.points[0].x == e1.points[1].x and e.points[0].y == e1.points[1].y and e.points[1].x == e1.points[0].x and e.points[1].y == e1.points[0].y:
                        flag1 = True
                
                flag2 = False
                e2 = trig1.edges[1] 
                for e in trig.edges :
                    if e.points[0].x == e2.points[0].x and e.points[0].y == e2.points[0].y and e.points[1].x == e2.points[1].x and e.points[1].y == e2.points[1].y:  
                        flag2 = True
                    elif e.points[0].x == e2.points[1].x and e.points[0].y == e2.points[1].y and e.points[1].x == e2.points[0].x and e.points[1].y == e2.points[0].y:
                        flag2 = True

                        
                flag3 = False
                e3 = trig1.edges[2] 
                for e in trig.edges :
                    if e.points[0].x == e3.points[0].x and e.points[0].y == e3.points[0].y and e.points[1].x == e3.points[1].x and e.points[1].y == e3.points[1].y:  
                        flag3 = True
                    elif e.points[0].x == e3.points[1].x and e.points[0].y == e3.points[1].y and e.points[1].x == e3.points[0].x and e.points[1].y == e3.points[0].y:
                        flag3 = True


                if flag1 and flag2 and flag3:
                    flag = True
            
            
            if flag:
                shared_tets.append(tet)
                return True , shared_tets   

    return False 

# Function 4:
def isContainPointsFromTrig(t1, t2): # check if two trigs are sharing a node
    for p1 in t1.points():
        for p2 in t2.points():
            if p1.x == p2.x and p1.y == p2.y:
                return True

    return False

# Function 5:
def createTetFromTrigAndPoint(trig, p):
    points = trig.points()
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]

    # e1 = Edge([p1, p2])
    # e2 = Edge([p2, p3])
    # e2i = Edge([p3, p2])
    # e3 = Edge([p3, p1])
    # e3i = Edge([p1, p3])
    # e4 = Edge([p, p1])
    # e5 = Edge([p, p3])
    # e5i = Edge([p3, p])
    # e6 = Edge([p2, p])

    # t1 = Triangle([e1, e2, e3]) #aristera
    # t2 = Triangle([e1, e6, e4]) #kato
    # t3 = Triangle([e6, e5, e2i]) #piso
    # t4 = Triangle([e4, e3i, e5i]) #brosta

    # e1 = Edge([edge.points[0], edge.points[1]])
    # e2 = Edge([edge.points[1], point])
    # e3 = Edge([point, edge.points[0]])
    # t = Triangle([e1, e2, e3])

    tet = Tetrahedron()
    tet.frompoints([p1,p2,p3,p])
   
    return tet


# Function 6:
def checkDelaunay(triangle):
    for e in triangle.edges:
        for t in e.trigs:
            if t == triangle:
                continue
            for p in t.points():
                if pointInsideCircumcircle(p, triangle):
                    print('Alert')
    return 1

# Function 7:
def calculateCircle(t):
    pnts = t.points()
    p1 = [pnts[0].x, pnts[0].y]
    p2 = [pnts[1].x, pnts[1].y]
    p3 = [pnts[2].x, pnts[2].y]

    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

#Find perpedicular line from 2 points
def perpendicular(point1, point2):
    m_x = (point1.x + point2.x) / 2
    m_y = (point1.y + point2.y) / 2

    if point2.x == point1.x:
        a = 0
        b = m_y
    elif point2.y == point1.y:
        a = None
        b = m_x
    else:
        slope = (point2.y - point1.y) / (point2.x - point1.x)
        a = -1 / slope
        b = m_y - a * m_x

    return a, b

#Find intersection point of 2 lines
def intersection(line1, line2):
    if line1[0] is None:
        if line2[0] is None:
            return None
        x0 = line1[1]
        y0 = line2[0] * x0 + line2[1]
    elif line2[0] is None:
        x0 = line2[1]
        y0 = line1[0] * x0 + line1[1]
    elif line1[0] == line2[0]:
        return None
    else:
        x0 = (line2[1] - line1[1]) / (line1[0] - line2[0])
        y0 = line1[0] * x0 + line1[1]

    return x0, y0

#Check if point is inside boundaries
def in_boundaries(this_point):
    if b2 <= this_point[0] <= b1:
        if b4 <= this_point[1] <= b3:
            return True
    return False

#Number of intersections of a line with any edge
def number_of_intersections(line, line_table):
    count = 0
    e1 = LineString([line[0], line[1]])
    for l in line_table:
        e2 = LineString([l[0], l[1]])
        if e1.intersects(e2):
            count += 1

    return count


# Find Delaunay Triangulation, exactly as in Wiki
def DelaunayTets(i):
    print("LOOP:", i)
    p = points[i]
    bad_tets = []
    for tet in tets:
        if pointInsideSphere(p, tet):  # first find all the triangles that are no longer valid due to the insertion
            bad_tets.append(tet)
    poly = []
    for b_t in bad_tets:
        for trig in b_t.triangles:
            copied_bad_tets = bad_tets[:] # remove from bad_tets the bad tet that we are investigating 
            copied_bad_tets.remove(b_t)
            flag  = isSharedTrig(trig, copied_bad_tets, b_t)
            if not flag:
                poly.append(trig)
    for b_t in bad_tets:
        tets.remove(b_t)
    for tr in poly:
        T = createTetFromTrigAndPoint(tr, p)
        tets.append(T)
    # Auto leipei kai isws ftaei poy den afaireitai to super trig
    # for each triangle in triangulation // done inserting points, now clean up
    #   if triangle contains a vertex from original super-triangle
    #       remove triangle from triangulation
    # return triangulation
    # plt.cla()

    # Draw points
    # np_points = np.array(list(map(lambda p: np.asarray([p.x, p.y]), points)))
    # plt.scatter(np_points[3:, 0], np_points[3:, 1], s=15)

    # Draw triangles and circles (output can be manipulated from the comments)
    #artists = []
    #for t in trigs[:]:
    #    trig_artist = t.toArtist()
    #    artists.append(trig_artist)
    #    plt.gca().add_patch(trig_artist)
    #    c = Circle()
    #    c.fromTriangle(t)
    #    circ_artist = c.toArtist()
    #    artists.append(circ_artist)
    #    plt.gca().add_artist(circ_artist) # Circle drawing

    

    
    return  tets
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dataset point input
arr = [(52, 12, 13.5), (13, 12, -2.6), (-21, 8, 2), (12, 10, 7), (13, 22, 6), (7, 11, -16), (-10, -5, 0), (-14, -3, 4), (20, 11, 8), (12, 4, 12)]
points = []
N=0 
for a in arr:
    points.append(Point(a[0],a[1],a[2]))
    N= N+1

# Copy path of Ski_Areas_NA.csv to paste below (the data can be manipulated manually to change the grid)
# filename = r'C:\Users\Dimitris\Documents\GitHub\Voronoi-Clustering\ProjectZoulf\airports - 50.csv' 


# points = []
# N=0
# with open(filename, 'r', encoding='utf8') as csvfile:
#     for line in csvfile:
#         separated = line.split(',')
#         # The points represented by the coordinates of the dataset are mostly in columns 6 and 7
#         # In some cases those coordinates are in columns 7 and 8, so we catch these exceptions
#         temp1 = float(separated[6])
#         temp2 = float(separated[7])
#         temp = [float(separated[6]), float(separated[7])]
#         '''
#         try:
#             temp1 = float(separated[5])
#             temp2 = float(separated[6])
#             temp = [float(separated[5]), float(separated[6])]
#         except ValueError:
#             temp1 = float(separated[7])
#             temp2 = float(separated[8])
#             temp = [float(separated[7]), float(separated[8])]
#         '''
#         N = N+1    # Number of points required for the plot/animation
#         #print(temp)     # Prints our point coordinates in the output console  
#         # Appends the scanned points into the point array as data of the Point Class
#         points.append(Point(temp1,temp2))

print('Number of points = ', N)

# # Manual user input:
# '''
# points = []
# N = int(input()) # Count of points
# for i in range(N):
#     xy = list(map(int, input().split(' ')[:2]))
#     points.append(Point(xy[0], xy[1]))
# '''

# # Automatic input (20 random points)
# '''
# N = 20 # Count of points
# points = list(map(lambda p: Point(p[0], p[1]), np.random.rand(N, 2)))
# for p in points:
#     p.x = p.x * 1.5
# '''

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Get points before super triangle
# X1 = [] 
# Y1 = []
# for z in points:
#     X1.append(z.x) 
#     Y1.append(z.y)

# Calculate SuperTetrahedron
super_tet = calculateSuperTetrahedron(points) 
tets = [super_tet]

stpoints =[]
for p in super_tet.points():
    stpoints.append(p)
    print("Printing at end",p.x, p.y, p.z)
 
# for t in super_tet.triangles:
#     print("\n")
#     for e in t.points():
#         print(e.x,e.y,e.z)


x1 = []
y1 = []
z1 = []
for p in points:
    x1.append(p.x)
    y1.append(p.y)
    z1.append(p.z)

# Draw points
fig = plt.figure()
axis = fig.add_subplot(111, projection='3d')
axis.scatter(x1,y1,z1)

# verts =[]

# for t in super_tet.triangles:
#     verts.append(t.points)
# axis.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

super_tet.painttet()

print('Running Delaunay Triangulation')

# s = Sphere()
# s.fromTetrahedron(super_tet)

# (xs,ys,zs) = s.paintsphere()
# axis.plot_wireframe(xs, ys, zs, color="r")

# s.paintsphere()
# print("sphere", s.x,s.y,s.z,s.radius)



# DelaunayTets(1)
# for i in range(1,N):
#     if 1<N:
#         DelaunayTets(i)
#     else: 
#         tets.append(DelaunayTets(i))

# counter= 0
# for t in trigs:
#     counter= counter +1

# print('Number of Delaunay triangles = ', counter)

# print('Drawing Voronoi Cells')

# # Draw Voronoi cells
# centersX = []
# centersY=[]
# v_edges = []
# for t in trigs:
#     flag3 = isContainPointsFromTrig(t,super_trig)
#     if t!= super_trig and not flag3:
#         for e in t.edges:
#             flag , vtrigs = isSharedEdge(e, trigs)
#             if flag:
#                 for t2 in vtrigs:
#                     flag2 = isContainPointsFromTrig(t2,super_trig)
#                     if not flag2 and t2!=t:
#                         c1 = Circle()
#                         c2 = Circle()
#                         c1.fromTriangle(t)
#                         c2.fromTriangle(t2)
#                         current_v_edge = [(c1.x , c1.y),(c2.x, c2.y)]
#                         v_edges.append(current_v_edge)
#                         #centre1 = Point(c1.x, c1.y)
#                         #centre2 = Point(c2.x, c2.y)
#                         #x = list(range(c1.x, c2.x))
#                         #y = list(range(c1.y, c2.y))
#                         x = [c1.x, c2.x]
#                         y = [c1.y, c2.y]
#                         centersX.append(c1.x)
#                         centersY.append(c1.y)
#                         plt.plot(x,y,'r')
#                         #e1 = Edge([centre1, centre2])
#                         #edge_artist = e1.toArtist()
#                         #artists.append(edge_artist)
#                         #plt.gca().add_patch(edge_artist)


# # Find Borders
# x2 = []
# y2 = []

# for i in centersX:
#     x2.append(i)
# for i in centersY:
#     y2.append(i)

# b1 = max(max(X1), max(x2))  #max x
# b2 = min(min(X1), min(x2))  #min x
# addx = abs(b2-b1)*0.1
# b1 = b1 + addx
# b2 = b2 - addx
# b3 = max(max(Y1), max(y2)) #max y
# b4 = min(min(Y1), min(y2)) #min y
# addy = abs(b3-b4)*0.1
# b3 = b3 + addy
# b4 = b4 - addy
# boundaries = [(0, b3), (0, b4), (None, b1), (None, b2)]

# xborders = [b1, b1, b2, b2, b1]
# yborders = [b3, b4, b4, b3, b3]
# plt.plot(xborders,yborders,'g')

# # Draw Voronoi semilines
# print('Drawing Voronoi Semi-lines')
# semi_lines= []
# for t in trigs:
#     flag3 = isContainPointsFromTrig(t,super_trig)
#     if not flag3:
#         for e in t.edges:
#             flag , vtrigs = isSharedEdge(e, trigs)
#             if flag:
#                 for t2 in vtrigs:
#                     flag2 = isContainPointsFromTrig(t2,super_trig)
#                     if flag2 and t2!=t:
#                         p = perpendicular(e.points[0],e.points[1])
#                         for b in boundaries:
#                             b_point = intersection(p,b)
#                             if b_point is not None:
#                                     if in_boundaries(b_point):
#                                         c1 = Circle()
#                                         c1.fromTriangle(t)
#                                         cp = c1.x, c1.y
#                                         semi_line = [cp, b_point]
#                                         if number_of_intersections(semi_line, v_edges) <= 2:
#                                             #if number_of_intersections(semi_line, semi_lines) <=2:
#                                             semi_lines.append(semi_line)



# # Draw semilines
# lc = mc.LineCollection(semi_lines, colors = 'r')
# ax.add_collection(lc)


# # def init(): # ??????
# #     np_points = np.array(list(map(lambda p: np.asarray([p.x, p.y]), points)))
# #     plt.scatter(np_points[:, 0], np_points[:, 1], s=15)
# #     return []



                 
# # ???


# #fanim = animation.FuncAnimation(
# #    fig, animate, init_func=init, frames=N + 3, interval=100, blit=True)


# #fanim.save('triangulation.gif', writer='pillow') 

# Outputs the results in a window
print('Results:')
plt.show()