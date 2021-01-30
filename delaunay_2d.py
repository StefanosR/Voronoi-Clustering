from sys import float_repr_style
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Info
# Algorithm: Delaunay Triangulation
# List triangulation
# Add triangle that envelops all the points in the triangulation list
# 1. Add point.
# 2. Find all triangles where the new point is in (bad triangles).
# 3. Find all edges that are between bad triangle and good triangle.
# 4. Construct new triangles with the edges and the new point.
# 5. Remove bad triangles.
# 6. Repeat until no more points can be added.
# 7. Remove all triangles that have a vertex from the super triangle.

# For transforming Delaunay -> Voronoi
# 1. Find all the circumcenters of the triangles. These are the voronoi points
# 2. Connect adjacent triangle circumcenters with edge.
# 3. For semilines, find the shared edge between a triangle with one that has 
#    a shared point with the super triangle. Create a semiline starting from the circumcenter of the triangle
#    perpendicular to the shared edge.
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Imported data will be assigned here as points
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

# Consists of the points and triangles of each edge (2)
class Edge:
    def __init__(self, points=[], trigs=[]):
        self.points = points
        self.trigs = trigs

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
    # Draw the triangle 
    def toArtist(self):
        points = np.array(list(map(lambda p: np.asarray([p.x, p.y]), self.points())))
        return plt.Polygon(points[:3, :], color='C0', alpha=0.8, fill=False, clip_on=True, linewidth=1)

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

# Function 1: Calculates the super triangle
def calculateSuperTriangle(points):
    # Find boundary box that contain all points
    p_min = Point(min(points, key=lambda p: p.x).x - 0.1,
                  min(points, key=lambda p: p.y).y - 0.1)
    p_max = Point(max(points, key=lambda p: p.x).x + 0.1,
                  max(points, key=lambda p: p.y).y + 0.1)

    a = p_max.x - p_min.x  # "distance" between max and min points
    b = p_max.y - p_min.y

    p1 = Point(p_min.x, p_min.y)
    p2 = Point(p_min.x, p_max.y + b)
    p3 = Point(p_max.x + a, p_min.y)

    points.insert(0, p1)
    points.insert(0, p2)
    points.insert(0, p3)

    e1 = Edge([p1, p2])
    e2 = Edge([p2, p3])
    e3 = Edge([p3, p1])

    t = Triangle([e1, e2, e3])
   
    return t

# Function 2: find if point is inside a circle
def pointInsideCircumcircle(p, t):
    pnts = t.points()

    p3 = pnts[0]
    p2 = pnts[1]
    p1 = pnts[2]

    x0 = p.x
    y0 = p.y
    x1 = p1.x
    y1 = p1.y
    x2 = p2.x
    y2 = p2.y
    x3 = p3.x
    y3 = p3.y

    ax_ = x1-x0
    ay_ = y1-y0
    bx_ = x2-x0
    by_ = y2-y0
    cx_ = x3-x0
    cy_ = y3-y0

    return (
        (ax_*ax_ + ay_*ay_) * (bx_*cy_-cx_*by_) -
        (bx_*bx_ + by_*by_) * (ax_*cy_-cx_*ay_) +
        (cx_*cx_ + cy_*cy_) * (ax_*by_-bx_*ay_)
    ) > 0

# Function 3: Finds if a certain edge is shared with any triangle
def isSharedEdge(edge, trigs):
    for t in trigs:
        for e in t.edges:  # check if the vertices of the inserted edge are same with those of the triangle
            if e.points[0].x == edge.points[0].x and e.points[0].y == edge.points[0].y and e.points[1].x == edge.points[1].x and e.points[1].y == edge.points[1].y:  
                return True
            elif e.points[0].x == edge.points[1].x and e.points[0].y == edge.points[1].y and e.points[1].x == edge.points[0].x and e.points[1].y == edge.points[0].y:
                return True

    return False

# Function 4: check if two trigs are sharing a node
def isContainPointsFromTrig(t1, t2): 
    for p1 in t1.points():
        for p2 in t2.points():
            if p1.x == p2.x and p1.y == p2.y:
                return True
    return False

# Function 5: create triangle from an edge and a point
def createTrigFromEdgeAndPoint(edge, point):
    e1 = Edge([edge.points[0], edge.points[1]])
    e2 = Edge([edge.points[1], point])
    e3 = Edge([point, edge.points[0]])
    t = Triangle([e1, e2, e3])

    return t

# Function 5: Whether a triangle contains a vertex from original super-triangle
def containsVertex(t1, t2):
    for v1 in t1.points():
        for v2 in t2.points():
            if v1.x == v2.x and v1.y == v2.y:
                return 0
            break    
        break
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dataset point input
# Copy path of Ski_Areas_NA.csv to paste below (the data can be manipulated manually to change the grid)
filename = r'C:\Users\Dimitris\Documents\GitHub\Voronoi-Clustering\airports - 50.csv' 

points = []
N=0
with open(filename, 'r', encoding='utf8') as csvfile:
    for line in csvfile:
        separated = line.split(',')
        # The points represented by the coordinates of the dataset are mostly in columns 6 and 7
        # In some cases those coordinates are in columns 7 and 8, so we catch these exceptions
        temp1 = float(separated[6])
        temp2 = float(separated[7])
        temp = [float(separated[6]), float(separated[7])]
        '''
        try:
            temp1 = float(separated[6])
            temp2 = float(separated[7])
            temp = [float(separated[6]), float(separated[7])]
        except ValueError:
            temp1 = float(separated[7])
            temp2 = float(separated[8])
            temp = [float(separated[7]), float(separated[8])]
        '''
        N +=1          # Number of points required for the plot/animation
        # print(temp)     # Prints our point coordinates in the output console  
        # Appends the scanned points into the point array as data of the Point Class
        points.append(Point(temp1,temp2))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Calculate SuperTriangle
super_trig = calculateSuperTriangle(points) 

trigs = [super_trig]

def init(): 
    np_points = np.array(list(map(lambda p: np.asarray([p.x, p.y]), points)))
    plt.scatter(np_points[:, 0], np_points[:, 1], s=15)
    return []

# Σύμφωνα με τον ψευδοκώδικα Wikipedia:
def animate(i):
    p = points[i]
    bad_trigs = []
    for t in trigs:                          # first find all the triangles that are no longer valid due to the insertion
        if pointInsideCircumcircle(p, t):  
            bad_trigs.append(t)
    poly = []
    for b_t in bad_trigs:                    # find the boundary of the polygonal hole
        for e in b_t.edges:
            copied_bad_trigs = bad_trigs[:]  # remove from bad_trigs the bad trig that we are investigating 
            copied_bad_trigs.remove(b_t)
            if not isSharedEdge(e, copied_bad_trigs):
                poly.append(e)
    for b_t in bad_trigs:
        trigs.remove(b_t)                    # remove them from the data structure
    for e in poly:
        T = createTrigFromEdgeAndPoint(e, p) # re-triangulate the polygonal hole
        trigs.append(T)
    # Κομμάτι που έλειπε
    for t in trigs:                          # for each triangle in triangulation // done inserting points, now clean up
        if containsVertex(t, super_trig):    # if triangle contains a vertex from original super-triangle
            trigs.remove(t)                  # remove triangle from triangulation
    # return triangulation
    plt.cla()

    # Draw points
    np_points = np.array(list(map(lambda p: np.asarray([p.x, p.y]), points)))
    plt.scatter(np_points[:, 0], np_points[:, 1], s=15)

    # Draw triangles and circles (output can be manipulated from the comments)
    artists = []
    for t in trigs[:]:
        trig_artist = t.toArtist()
        artists.append(trig_artist)
        plt.gca().add_patch(trig_artist)
        c = Circle()
        c.fromTriangle(t)
        circ_artist = c.toArtist()
        artists.append(circ_artist)
        plt.gca().add_artist(circ_artist) # Circle drawing
    return artists

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

fanim = animation.FuncAnimation(
fig, animate, init_func=init, frames=N + 3, interval=100, blit=True)

# Saves the results in gif:

# fanim.save('triangulation.gif', writer='pillow') 

# Outputs the results in a window

plt.show()