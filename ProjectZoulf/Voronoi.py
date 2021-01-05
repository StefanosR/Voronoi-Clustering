import math
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from shapely.geometry import LineString


class VoronoiSite:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.voronoi_points = []
        self.perpendiculars = []
        self.vectors = []
        self.semi_lines = []
        self.has_cell = False


def in_boundaries(this_point):
    if b2 <= this_point[0] <= b1:
        if b4 <= this_point[1] <= b3:
            return True
    return False


def distance(site, point):
    if type(site) == type(point):
        d = math.sqrt((site.x - point.x) ** 2 + (site.y - point.y) ** 2)

    else:
        d = math.sqrt((site.x - point[0]) ** 2 + (site.y - point[1]) ** 2)

    return d


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


def find_center(point1, point2, point3):
    p1 = perpendicular(point1, point2)
    p2 = perpendicular(point2, point3)
    center = intersection(p1, p2)

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
    m = [x, y]
    d1 = distance(this_site, m)

    for site in sites_table:
        if site != this_site:
            d2 = distance(site, m)
            if round(d1, 10) == round(d2, 10):
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

                        perpendicular1 = perpendicular(this_site, site1)
                        perpendicular2 = perpendicular(this_site, site2)
                        cur_center = find_center(this_site, site1, site2)

                        if cur_center is not None:
                            cur_center = [round(cur_center[0], 12), round(cur_center[1], 12)]

                            radius = distance(this_site, cur_center)

                            if not points_in_circle(cur_center, radius, sites_table):

                                # if cur_center not in this_site.voronoi_points:
                                this_site.voronoi_points.append(cur_center)

                                if perpendicular1 not in this_site.perpendiculars:
                                    this_site.perpendiculars.append(perpendicular1)

                                if perpendicular2 not in this_site.perpendiculars:
                                    this_site.perpendiculars.append(perpendicular2)

                        else:
                            d1 = distance(this_site, site1)
                            d2 = distance(this_site, site2)
                            c = [this_site.x, this_site.y]

                            if d1 < d2:
                                if not points_in_circle(c, d1 / 2, sites_table):
                                    if perpendicular1 not in this_site.perpendiculars:
                                        this_site.perpendiculars.append(perpendicular1)
                            elif d2 < d1:
                                if not points_in_circle(c, d2 / 2, sites_table):
                                    if perpendicular2 not in this_site.perpendiculars:
                                        this_site.perpendiculars.append(perpendicular2)

                            elif d1 == d2:
                                if not points_in_circle(c, d2 / 2, sites_table):

                                    if perpendicular1 not in this_site.perpendiculars:
                                        this_site.perpendiculars.append(perpendicular1)

                                    if perpendicular2 not in this_site.perpendiculars:
                                        this_site.perpendiculars.append(perpendicular2)


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

                            for b in boundaries:
                                other_point = intersection(p, b)
                                if other_point is not None:
                                    if in_boundaries(other_point):
                                        semi_line = [point, other_point]
                                        if number_of_intersections(semi_line, all_vectors) <= 2:
                                            this_site.semi_lines.append(semi_line)


# --------------------------------------Main-------------------------------------Main-----------------------------------

filename = r'C:\Users\Στέφανος Ρήγας\Desktop\Fast Projects\Voronoi Projects\ZoulfosVoronoi\airports - 50.csv'
arr = []
my_sites = []
all_vectors = []

with open(filename, 'r', encoding='utf8') as csvfile:
    for line in csvfile:
        separated = line.split(',')

        # Σχεδόν σε όλες τις καταχωρήσεις του dataset, οι συντεταγμένες είναι στις στήλες 6,7
        # εκτός απο ελάχιστες καταχωρήσεις που είναι στις 7,8 οπότε βάλαμε exception
        try:
            temp = [float(separated[6]), float(separated[7])]
        except ValueError:
            temp = [float(separated[7]), float(separated[8])]
        arr.append(temp)

print(arr)
for s in arr:
    x = VoronoiSite(s[0], s[1])
    my_sites.append(x)

find_voronoi_points(my_sites)

find_voronoi_cells(my_sites)
for site in my_sites:
    for vec in site.vectors:
        if vec not in all_vectors:
            all_vectors.append(vec)

x1 = []
y1 = []
x2 = []
y2 = []
for s in my_sites:
    x1.append(s.x)
    y1.append(s.y)
    for v in s.voronoi_points:
        x2.append(v[0])
        y2.append(v[1])

b1 = max(max(x1), max(x2))
b2 = min(min(x1), min(x2))
b3 = max(max(y1), max(y2))
b4 = min(min(y1), min(y2))
boundaries = [(0, b3), (0, b4), (None, b1), (None, b2)]

find_voronoi_semi_lines(my_sites)
for site in my_sites:
    for sl in site.semi_lines:
        if sl not in all_vectors:
            all_vectors.append(sl)

all_vectors.append([[b1, b3], [b1, b4]])
all_vectors.append([[b2, b3], [b2, b4]])
all_vectors.append([[b1, b3], [b2, b3]])
all_vectors.append([[b2, b4], [b1, b4]])

# -----------------------------------Graphics----------------------------------Graphics---------------------------------

lc = mc.LineCollection(all_vectors)
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.margins(0.1)

plt.plot(x1, y1, "o")
plt.plot(x2, y2, "o")

plt.show()
