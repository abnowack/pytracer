import numpy as np
import matplotlib.pyplot as plt
import math2d


class Detector(object):
    """
    Base Class for drawing detector on display and calculate detector segments
    """

    def calculate_segments(self, nbins):
        """
        Create nbin segments from the detector parameters

        Parameters
        ----------
        nbins : int
            Number of linear segments to represent the detector

        Returns
        -------
        out : (N, 2, 2) ndarray
            array of segments
        """
        raise NotImplementedError

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
        self.center = center
        self.width = width
        self.angle = angle

        self.segments = self.calculate_segments(nbins)
    
    def calculate_segments(self, nbins):
        points = np.zeros((nbins+1, 2))
        points[:, 1] = np.linspace(-self.width / 2., self.width / 2., nbins+1)
        rot = math2d.angle_matrix(self.angle)
        points = np.dot(points, rot)
        points[:, 0] += self.center[0]
        points[:, 1] += self.center[1]

        return math2d.create_segments_from_points(points)


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
        self.start_angle = start_angle
        self.end_angle = end_angle
        
        self.segments = self.calculate_segments(nbins)

    def calculate_segments(self, nbins):
        angles = np.linspace(self.end_angle, self.start_angle, nbins + 1) * np.pi / 180.
        points = np.zeros((nbins+1, 2), dtype=np.float32)
        points[:, 0] = self.radius * np.cos(angles) + self.center[0]
        points[:, 1] = self.radius * np.sin(angles) + self.center[1]

        return math2d.create_segments_from_points(points)