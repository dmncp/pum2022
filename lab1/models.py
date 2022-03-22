import numpy as np


class Hypersphere:
    def __init__(self, p1, p2):
        self.central_point = p1
        self.second_point = p2
        self.r = self.radius()

    def radius(self):
        return Vector(self.central_point, self.second_point).distance()

    def point_inside(self, point):
        if np.array_equal(self.central_point, point):
            return True
        return Vector(self.central_point, point).distance() < self.r


class Vector:
    def __init__(self, p1, p2):
        self.coordinates = p1 - p2

    def distance(self):  # Euclidean distance
        return np.linalg.norm(self.coordinates)

    def angle(self, other):  # angle between two vectors
        inner = np.inner(self.coordinates, other.coordinates)
        norms = np.linalg.norm(self.coordinates) * np.linalg.norm(other.coordinates)
        rad = np.arccos(np.clip(inner/norms, -1.0, 1.0))
        return np.rad2deg(rad)
