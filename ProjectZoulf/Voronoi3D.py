from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.pyplot as plt
#from shapely.geometry import LineString


class VoronoiSite:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.voronoi_points = []
        self.perpendiculars = []
        self.vectors = []
        self.semi_lines = []
        self.has_cell = False


def distance(this_site, point):
    if type(this_site) == type(point):
        d = math.sqrt((this_site.x - point.x) ** 2 + (this_site.y - point.y) ** 2 + (this_site.z - point.z) ** 2)

    else:
        d = math.sqrt((this_site.x - point[0]) ** 2 + (this_site.y - point[1]) ** 2 + (this_site.z - point[2]) ** 2)

    return d


def intersection(plane1, plane2, plane3):

    angle12 = calcAngle(plane1, plane2)
    angle13 = calcAngle(plane1, plane3)
    angle23 = calcAngle(plane2, plane3)

    if angle12 == 0 or angle13 == 0 or angle23 == 0 or angle12 == 180 or angle13 == 180 or angle23 == 180:

       return None

    else:

        detZ = plane1[0]*plane2[1] - plane1[1]*plane2[0]

        if detZ != 0:
            det0 = plane1[1]*plane2[3] - plane2[1]*plane1[3]
            det1 = plane1[1]*plane2[2] - plane2[1]*plane1[2]

            x = det0 / detZ
            xPar = det1 / detZ

            det0 = plane2[0]*plane1[3] - plane1[0]*plane2[3]
            det1 = plane2[0]*plane1[2] - plane2[2]*plane1[0]

            y = det0 / detZ
            yPar = det1 / detZ

            #print(x, xPar, y, yPar)

            #check if the line is parallel to the 3rd plane
            x1 = x + xPar*1
            y1 = y + yPar*1
            z1 = 1

            x2 = x + xPar*2
            y2 = y + yPar*2
            z2 = 2

            directionVector = [x2 - x1, y2 - y1, z2 - z1]
            #print(directionVector)
            inProd = plane3[0]*xPar + plane3[1]*yPar + plane3[2]
            round(inProd, 14)
            print("inProd", inProd)
            if inProd == 0:

                return None

            else:

                t = -(plane3[0]*x + plane3[1]*y + plane3[3])/(plane3[0]*xPar + plane3[1]*yPar + plane3[2])

                x0 = round(x + xPar*t, 5)
                y0 = round(y + yPar*t, 5)
                z0 = round(t, 5)

        else:

            detX = plane1[1]*plane2[2] - plane1[2]*plane2[1]

            det0 = plane1[2]*plane2[3] - plane1[3]*plane2[2]
            det1 = plane1[2]*plane2[0] - plane1[0]*plane2[2]

            y = det0 / detX
            yPar = det1 / detX

            det0 = plane1[3]*plane2[1] - plane1[1]*plane2[3]
            det1 = plane1[0]*plane2[1] - plane1[1]*plane2[0]

            z = det0 / detX
            zPar = det1 / detX

            #check if the line is parallel to the 3rd plane
            x1 = 1
            y1 = y + yPar*1
            z1 = z + zPar*1

            x2 = 2
            y2 = y + yPar*2
            z2 = z + zPar*2

            directionVector = [x2 - x1, y2 - y1, z2 - z1]
            inProd = plane3[0] + plane3[1]*yPar + plane3[2]*zPar
            round(inProd, 14)
            print("inprod", inProd)
            if inProd == 0:

                return None

            else:

                t = -(plane3[1]*y + plane3[2]*z + plane3[3]) / (plane3[0] + plane3[1]*yPar + plane3[2]*zPar)

                x0 = round(t, 5)
                y0 = round(y + yPar*t, 5)
                z0 = round(z + zPar*t, 5)

    return x0, y0, z0


def calcAngle(plane1, plane2):

    prod = plane1[0]*plane2[0] + plane1[1]*plane2[1] + plane1[2]*plane2[2]
    root1 = math.sqrt(plane1[0]*plane1[0] + plane1[1]*plane1[1] + plane1[2]*plane1[2])
    root2 = math.sqrt(plane2[0]*plane2[0] + plane2[1]*plane2[1] + plane2[2]*plane2[2])
    cos = round(prod / (root1 * root2), 2)
    angle = math.degrees(math.acos(cos))

    return angle


def perpendicular(point1, point2):

    mx = (point1.x + point2.x)/2
    my = (point1.y + point2.y)/2
    mz = (point1.z + point2.z)/2

    v1 = point1.x - point2.x
    v2 = point1.y - point2.y
    v3 = point1.z - point2.z
    v4 = -v1*mx -v2*my -v3*mz

    plane = [v1, v2, v3, v4]

    return plane


def find_center(point1, point2, point3):
    print("p1", point1, "p2", point2, "p3", point3)



    p1 = perpendicular(point1, point2)
    p2 = perpendicular(point1, point3)
    p3 = perpendicular(point2, point3)

    center = intersection(p1, p2, p3)
    print("plane", p1, p2, p3)
    return center


def points_in_circle(center, radius, sites_table):
    flag = False
    for site in sites_table:
        d = distance(site, center)

        if round(d, 5) < round(radius, 5):
            flag = True
    return flag


def is_vector_perpendicular(this_vector, this_site, sites_table):
    x = (this_vector[0][0] + this_vector[1][0]) / 2
    y = (this_vector[0][1] + this_vector[1][1]) / 2
    z = (this_vector[0][2] + this_vector[1][2]) / 2

    m = [x, y, z]
    d1 = distance(this_site, m)

    for site in sites_table:
        if site != this_site:
            d2 = distance(site, m)
            if round(d1, 5) == round(d2, 5):
                return True
    return False


def number_of_intersections(line, line_table):
    count = 0
    e1 = LineString([line[0], line[1]])
    for l in line_table:
        e2 = LineString([l[0], l[1]])
        if e1.intersects(e2):
            count += 1

    return count


def find_voronoi_points(sites_table):
    for this_site in sites_table:
        sites_tested = [this_site]

        for site1 in sites_table:
            if site1 not in sites_tested:
                sites_tested.append(site1)

                for site2 in sites_table:
                    if site2 not in sites_tested:

                        cur_center = find_center(this_site, site1, site2)

                        if cur_center is not None:
                            cur_center = [round(cur_center[0], 12), round(cur_center[1], 12), round(cur_center[2], 12)]

                            radius = distance(this_site, cur_center)

                            if not points_in_circle(cur_center, radius, sites_table):

                                #if cur_center not in this_site.voronoi_points:
                                this_site.voronoi_points.append(cur_center)


def find_voronoi_cells(sites_table):
    for this_site in sites_table:
        points_checked = []

        for voronoi_point1 in this_site.voronoi_points:
            points_checked.append(voronoi_point1)

            for voronoi_point2 in this_site.voronoi_points:

                if voronoi_point2 not in points_checked:
                    vector = [voronoi_point1, voronoi_point2]
                    rev_vector = [voronoi_point2, voronoi_point1]

                    if vector not in this_site.vectors and rev_vector not in this_site.vectors:
                        if is_vector_perpendicular(vector, this_site, sites_table):
                            this_site.vectors.append(vector)

        for point in this_site.voronoi_points:
            count = 0
            for vector in this_site.vectors:
                if vector[0] == point or vector[1] == point:
                    count += 1

            this_site.has_cell = count == 2


def find_voronoi_semi_lines(sites_table):
    for this_site in sites_table:
        if not this_site.has_cell:

            for neighbor_site in sites_table:
                if not neighbor_site.has_cell and this_site != neighbor_site:

                    for point in this_site.voronoi_points:

                        if point in neighbor_site.voronoi_points:

                            p = perpendicular(this_site, neighbor_site)

                            other_point = intersection(p, b)
                            if other_point is not None:
                                semi_line = [point, other_point]
                                if number_of_intersections(semi_line, all_vectors) <= 2:
                                    this_site.semi_lines.append(semi_line)


# --------------------------------------Main-------------------------------------Main-----------------------------------

arr = [(52, 12, 13.5), (13, 12, -2.6), (-21, 8, 2), (12, 10, 7), (13, 22, 6), (7, 11, -16), (-10, -5, 0), (-14, -3, 4), (20, 11, 8), (12, 4, 12)]
my_sites = []
all_vectors = []


for s in arr:
    x = VoronoiSite(s[0], s[1], s[2])
    my_sites.append(x)

find_voronoi_points(my_sites)
find_voronoi_cells(my_sites)

for site in my_sites:
    for vec in site.vectors:
        if vec not in all_vectors:
            all_vectors.append(vec)

x1 = []
y1 = []
z1 = []

x2 = []
y2 = []
z2 = []

for s in my_sites:
    x1.append(s.x)
    y1.append(s.y)
    z1.append(s.z)

    for v in s.voronoi_points:
        x2.append(v[0])
        y2.append(v[1])
        z2.append(v[2])

# -----------------------------------Graphics----------------------------------Graphics---------------------------------
fig = plt.figure()

axis = fig.add_subplot(111, projection='3d')

X = ([x1],[x1])
Y = ([y1],[y1])
Z = ([z1],[z1])


print(all_vectors)


axis.scatter(x1,y1,z1)
axis.scatter(x2,y2,z2)

plt.show()