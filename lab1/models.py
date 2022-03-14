import math


# converts radians to degrees
def deg(radians):
    return (radians * 180) / math.pi


class Hypersphere:
    def __init__(self, p1, p2):
        self.central_point = p1
        self.second_point = p2
        self.r = self.radius()

    def radius(self):
        return Vector(self.central_point, self.second_point).distance()

    def point_inside(self, point):
        if self.central_point == point:
            return True
        return Vector(self.central_point, point).distance() < self.r


class Vector:
    def __init__(self, p1, p2):
        self.start_point = p1
        self.end_point = p2
        self.n = len(self.start_point)  # number of dimensions
        self.coordinates = self.get_coordinates()

    def get_coordinates(self):
        coord = []
        for x in range(self.n):
            coord.append(self.end_point[x] - self.start_point[x])
        return coord

    def distance(self):  # Euclidean distance
        squares_sum = 0
        for x in range(self.n):
            squares_sum += self.coordinates[x]**2

        return math.sqrt(squares_sum)

    def angle(self, other):  # angle between two vectors
        scalar_product = 0
        for x in range(self.n):
            scalar_product += self.coordinates[x] * other.coordinates[x]
        dist_prod = self.distance() * other.distance()

        return deg(math.acos(round(scalar_product / dist_prod, 3)))
