import numpy as np
import math2d


class Source(object):
    def __init__(self, x, y, angle=0.):
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