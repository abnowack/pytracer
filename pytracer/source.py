import numpy as np
from . import math2d


class Source(object):
    def __init__(self, x, y, angle=0):
        self._init_pos = np.array([x, y])
        self._init_angle = angle

        self.angle = angle
        self.pos = self._init_pos
        self.render()

    def render(self):
        rot = math2d.angle_matrix(self.angle)
        self.pos = np.dot(self._init_pos, rot)

    def rotate(self, angle):
        self.angle = angle
        self.render()

    def emit(self, start_angle, end_angle, n):
        angles = np.linspace(start_angle, end_angle, n)
        angles *= np.pi / 180

        end = np.ones((n, 2))
        end[:, 0] = np.cos(np.pi * self.angle / 180 - angles)
        end[:, 1] = np.sin(np.pi * self.angle / 180 - angles)

        return end
