import matplotlib.pyplot as plt
import numpy as np

import math2d


class Detector(object):
    """
    Base Class for drawing detector on display and calculate detector segments
    """

    def draw(self, show_normal=False, color='green'):
        """
        Display detector by plotting the segments on a matplotlib canvas

        Parameters
        ----------
        show_normal : bool
            Display normal vectors of detector segments
        color : str
            Color of detector segments to display
        """
        for i, segment in enumerate(self.segments):
            plt.plot(segment[:, 0], segment[:, 1], color=color)
            if show_normal:
                normal = math2d.normal(segment)
                center = math2d.center(segment)
                plt.arrow(center[0], center[1], normal[0], normal[1], width=0.01, color=color)


class DetectorPlane(Detector):
    def __init__(self, center, width, nbins, angle=0):
        """
        Linear Detector Plane

        Parameters
        ----------
        center : (2) ndarray
            [center_x, center_y] of detector plane midpoint
        width : float
            Full width of detector plane
        nbins : int
            Number of segments to break up detector plane
        angle : float
            Rotation angle of detector with respect to (0, 1) y-axis
        """
        self._center = center
        self.center = self._center
        self._width = width
        self._init_angle = angle
        self.angle = self._init_angle
        self._nbins = nbins
        self.segments = None

        self.render()

    @property
    def nbins(self):
        return self._nbins

    @nbins.setter
    def nbins(self, value):
        self._nbins = value
        self.render()

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self.render()

    def render(self):
        points = np.zeros((self.nbins + 1, 2))
        points[:, 1] = np.linspace(-self._width / 2., self._width / 2., self.nbins + 1)
        rot = math2d.angle_matrix(self.angle)
        points = np.dot(points, rot)

        self.center = np.dot(self._center, rot)

        points[:, 0] += self.center[0]
        points[:, 1] += self.center[1]

        self.segments = math2d.create_segments_from_points(points)

    def rotate(self, angle):
        self.angle = angle
        self.render()


class DetectorArc(Detector):
    def __init__(self, center, radius, start_angle, end_angle, nbins):
        """
        Circular segment detector plane. Defined in terms of a circle with center and radius

        Parameters
        ----------
        center : (2) ndarray
            Center of circle defining detector arc
        radius : float
            Radius of circle defining detector arc
        start_angle : float
            Start angle of detector arc along circle
        end_angle : float
            End angle of detector arc along circle
        nbins : int
            Number of linear segments to approximate detector plane
        """
        self.center = center
        self.radius = radius
        self._init_start_angle = start_angle
        self._init_end_angle = end_angle
        self.angle = 0.
        self.start_angle, self.end_angle = self._init_start_angle, self._init_end_angle
        self._nbins = nbins
        self.segments = None

        self.render()

    @property
    def nbins(self):
        return self._nbins

    @nbins.setter
    def nbins(self, value):
        self._nbins = value
        self.render()

    def render(self):
        angles = np.linspace(self.end_angle, self.start_angle, self.nbins + 1) * np.pi / 180.
        points = np.zeros((self.nbins + 1, 2), dtype=np.float32)
        points[:, 0] = self.radius * np.cos(angles) + self.center[0]
        points[:, 1] = self.radius * np.sin(angles) + self.center[1]

        self.segments = math2d.create_segments_from_points(points)

    def rotate(self, angle):
        self.start_angle = self._init_start_angle + angle
        self.end_angle = self._init_end_angle + angle
        self.render()
