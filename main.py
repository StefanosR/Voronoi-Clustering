import numpy as np
import pylab as plt
import matplotlib.animation as animation
from matplotlib import collections as mc
from shapely.geometry import Point, LineString



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
# class Point:
#     def __init__(self, x=0, y=0):
#         self.x = x
#         self.y = y

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

# Function 2:
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
    shared_trigs = []
    for t in trigs:
        for e in t.edges:  # check if the vertices of the inserted edge are same with those of the triangle
            if e.points[0].x == edge.points[0].x and e.points[0].y == edge.points[0].y and e.points[1].x == edge.points[1].x and e.points[1].y == edge.points[1].y:  
                shared_trigs.append(t)
                return True , shared_trigs
            elif e.points[0].x == edge.points[1].x and e.points[0].y == edge.points[1].y and e.points[1].x == edge.points[0].x and e.points[1].y == edge.points[0].y:
                shared_trigs.append(t)
                return True , shared_trigs

    return False

# Function 4:
def isContainPointsFromTrig(t1, t2): # check if two trigs are sharing a node
    for p1 in t1.points():
        for p2 in t2.points():
            if p1.x == p2.x and p1.y == p2.y:
                return True

    return False

# Function 5:
def createTrigFromEdgeAndPoint(edge, point):
    e1 = Edge([edge.points[0], edge.points[1]])
    e2 = Edge([edge.points[1], point])
    e3 = Edge([point, edge.points[0]])
    t = Triangle([e1, e2, e3])

    return t

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
def DelaunayTrigs(i):
    p = points[i]
    bad_trigs = []
    for t in trigs:
        if pointInsideCircumcircle(p, t):  # first find all the triangles that are no longer valid due to the insertion
            bad_trigs.append(t)
    poly = []
    for b_t in bad_trigs:
        for e in b_t.edges:
            copied_bad_trigs = bad_trigs[:] # remove from bad_trigs the bad trig that we are investigating 
            copied_bad_trigs.remove(b_t)
            flag  = isSharedEdge(e, copied_bad_trigs)
            if not flag:
                poly.append(e)
    for b_t in bad_trigs:
        trigs.remove(b_t)
    for e in poly:
        T = createTrigFromEdgeAndPoint(e, p)
        trigs.append(T)
    # Auto leipei kai isws ftaei poy den afaireitai to super trig
    # for each triangle in triangulation // done inserting points, now clean up
    #   if triangle contains a vertex from original super-triangle
    #       remove triangle from triangulation
    # return triangulation
    plt.cla()

    # Draw points
    np_points = np.array(list(map(lambda p: np.asarray([p.x, p.y]), points)))
    plt.scatter(np_points[3:, 0], np_points[3:, 1], s=15)

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

    

    
    return  trigs
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Dataset point input

# Copy path of Ski_Areas_NA.csv to paste below (the data can be manipulated manually to change the grid)
filename = r'C:\Users\Dimitris\Documents\GitHub\Voronoi-Clustering\ProjectZoulf\airports - 100.csv' 

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
        N = N+1    # Number of points required for the plot/animation
        #print(temp)     # Prints our point coordinates in the output console  
        # Appends the scanned points into the point array as data of the Point Class
        points.append(Point(temp1,temp2))

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
X1 = [] 
Y1 = []
for z in points:
    X1.append(z.x) 
    Y1.append(z.y)

# Calculate SuperTriangle
super_trig = calculateSuperTriangle(points) 
trigs = [super_trig]





fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal') # Αυτό βασικά μπορούμε να το κρατήσουμε

print('Running Delaunay Triangulation')

for i in range(1,N + 3):
    if 1<N+3:
        DelaunayTrigs(i)
    else: 
        trigs.append(DelaunayTrigs(i))

counter= 0
for t in trigs:
    counter= counter +1

print('Number of Delaunay triangles = ', counter)

print('Drawing Voronoi Cells')

# Draw Voronoi cells
centersX = []
centersY=[]
v_edges = []
for t in trigs:
    flag3 = isContainPointsFromTrig(t,super_trig)
    if t!= super_trig and not flag3:
        for e in t.edges:
            flag , vtrigs = isSharedEdge(e, trigs)
            if flag:
                for t2 in vtrigs:
                    flag2 = isContainPointsFromTrig(t2,super_trig)
                    if not flag2 and t2!=t:
                        c1 = Circle()
                        c2 = Circle()
                        c1.fromTriangle(t)
                        c2.fromTriangle(t2)
                        current_v_edge = [(c1.x , c1.y),(c2.x, c2.y)]
                        v_edges.append(current_v_edge)
                        #centre1 = Point(c1.x, c1.y)
                        #centre2 = Point(c2.x, c2.y)
                        #x = list(range(c1.x, c2.x))
                        #y = list(range(c1.y, c2.y))
                        x = [c1.x, c2.x]
                        y = [c1.y, c2.y]
                        centersX.append(c1.x)
                        centersY.append(c1.y)
                        plt.plot(x,y,'r')
                        #e1 = Edge([centre1, centre2])
                        #edge_artist = e1.toArtist()
                        #artists.append(edge_artist)
                        #plt.gca().add_patch(edge_artist)


# Find Borders
x2 = []
y2 = []

for i in centersX:
    x2.append(i)
for i in centersY:
    y2.append(i)

b1 = max(max(X1), max(x2))  #max x
b2 = min(min(X1), min(x2))  #min x
addx = abs(b2-b1)*0.1
b1 = b1 + addx
b2 = b2 - addx
b3 = max(max(Y1), max(y2)) #max y
b4 = min(min(Y1), min(y2)) #min y
addy = abs(b3-b4)*0.1
b3 = b3 + addy
b4 = b4 - addy
boundaries = [(0, b3), (0, b4), (None, b1), (None, b2)]

xborders = [b1, b1, b2, b2, b1]
yborders = [b3, b4, b4, b3, b3]
plt.plot(xborders,yborders,'g')

# Draw Voronoi semilines
print('Drawing Voronoi Semi-lines')
semi_lines= []
for t in trigs:
    flag3 = isContainPointsFromTrig(t,super_trig)
    if not flag3:
        for e in t.edges:
            flag , vtrigs = isSharedEdge(e, trigs)
            if flag:
                for t2 in vtrigs:
                    flag2 = isContainPointsFromTrig(t2,super_trig)
                    if flag2 and t2!=t:
                        p = perpendicular(e.points[0],e.points[1])
                        for b in boundaries:
                            b_point = intersection(p,b)
                            if b_point is not None:
                                    if in_boundaries(b_point):
                                        c1 = Circle()
                                        c1.fromTriangle(t)
                                        cp = c1.x, c1.y
                                        semi_line = [cp, b_point]
                                        if number_of_intersections(semi_line, v_edges) <= 2:
                                            #if number_of_intersections(semi_line, semi_lines) <=2:
                                            semi_lines.append(semi_line)



# Draw semilines
lc = mc.LineCollection(semi_lines, colors = 'r')
ax.add_collection(lc)


# def init(): # ??????
#     np_points = np.array(list(map(lambda p: np.asarray([p.x, p.y]), points)))
#     plt.scatter(np_points[:, 0], np_points[:, 1], s=15)
#     return []



                 
# ???


#fanim = animation.FuncAnimation(
#    fig, animate, init_func=init, frames=N + 3, interval=100, blit=True)


#fanim.save('triangulation.gif', writer='pillow') 

# Outputs the results in a window
print('Results:')
plt.show()