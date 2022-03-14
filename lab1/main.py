import random

from models import *
from viewer import *


# elements average
def avg(elements):
    return sum(elements) / len(elements)


# standard deviation
def std(elements):
    avg_val = avg(elements)
    return math.sqrt(sum([(x - avg_val)**2 for x in elements]) / len(elements))


# returns random point (vector)
def point(n):
    return [random.random() for _ in range(n)]  # n = number of dimensions (point vector size)


# get one point from points list
def get_point():
    return points[random.randint(0, len(points) - 1)]


# get pair of points from points list
def get_pair():
    point1 = get_point()
    point2 = get_point()
    while point1 == point2:
        point2 = get_point()
    return point1, point2


# appends points to list of points
def draw_points(n):
    for p in range(no_points):
        x = point(n)
        while x in points:
            x = point(n)
        points.append(x)


# appends angle between two vectors in degrees to list
def calculate_angle():
    point1, point2 = get_pair()
    point3, point4 = get_pair()
    vector1 = Vector(point1, point2)
    vector2 = Vector(point3, point4)
    angles.append(vector1.angle(vector2))


# appends ratio of points inside hypersphere to points outside
def calculate_ratio():
    point1, point2 = get_pair()
    sphere = Hypersphere(point1, point2)
    points_inside = 0

    for p in points:
        if sphere.point_inside(p):
            points_inside += 1
    ratios.append((points_inside / no_points) * 100)


# appends ratio of difference in distances to avg of distances
def calculate_distance_to_avg():
    point1 = get_point()
    point2, point3 = get_pair()
    dist1 = Vector(point1, point2).distance()
    dist2 = Vector(point1, point3).distance()

    distances.append(abs(dist1 - dist2) / avg([dist1, dist2]) * 100)


def run_test_angle(max_dim=500, no_tests=1000):
    angles_avg = []
    angles_std = []
    for d in range(2, max_dim):
        # first of all we have to draw points
        draw_points(d)

        for _ in range(no_tests):
            calculate_angle()
        angles_avg.append(avg(angles))
        angles_std.append(std(angles))
        clear_lists()

        print(f'{d}: Angle: {angles_avg[-1]} deg.')

    plot_angles([x for x in range(2, max_dim)], angles_avg, angles_std, max_dim)


def run_test_ratio(max_dim=500, no_tests=100):
    ratios_avg = []
    ratios_std = []
    for d in range(2, max_dim):
        # first of all we have to draw points
        draw_points(d)

        for _ in range(no_tests):
            calculate_ratio()
        ratios_avg.append(avg(ratios))
        ratios_std.append(std(ratios))
        clear_lists()

        print(f'{d} Inside/Outside ratio: {ratios_avg[-1]}%')

    plot_ratios([x for x in range(2, max_dim)], ratios_avg, ratios_std, max_dim)


def run_test_dist(max_dim=500, no_tests=1000):
    dist_avg = []
    dist_std = []
    for d in range(2, max_dim):
        # first of all we have to draw points
        draw_points(d)

        for _ in range(no_tests):
            calculate_distance_to_avg()
        dist_avg.append(avg(distances))
        dist_std.append(std(distances))
        clear_lists()

        print(f'{d} Difference/Avg: {dist_avg[-1]}%')
    plot_distances([x for x in range(2, max_dim)], dist_avg, dist_std, max_dim)


def clear_lists():
    points.clear()
    angles.clear()
    ratios.clear()
    distances.clear()


if __name__ == "__main__":
    no_points = 100

    points = []  # generated points list
    angles = []  # angles between two vectors (len(list) == no_tests)
    ratios = []  # list of ratios of points inside the hypersphere to points outside the hypersphere
    distances = []  # a list of ratios of the difference in distance between two points to the mean of these distances

    # we have to draw two pairs of points, so four points from hypercube points list
    run_test_angle()

    # next step - get ratios of points inside hypersphere to points outside
    run_test_ratio()

    # What % of the average of the two distances is the difference between them?
    run_test_dist()
