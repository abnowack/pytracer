import numpy as np
import math2d


def continuous_path_order(segments):
    """
    Reorder segments such that each segment is connected to each other in order
    [seg_1_2, seg_2_3, seg_3_4, ... seg_n-1_n, seg_n_1]

    Parameters
    ----------
    segments : (N, 2, 2) ndarray
        Array of N segments unordered

    Returns
    -------
    out : (N, 2, 2) ndarray
        Array of original segments reordered in contiguous path order

    """
    new_indices = []
    next_index = 0

    for i, segment in enumerate(segments):
        if next_index not in new_indices:
            new_indices.append(next_index)
        else:
            others = np.setdiff1d(segments, segments[new_indices])
            next_index = np.argwhere(segments[others[0]] == segments)
            new_indices.append(next_index)

        next_index = np.argwhere(segments[new_indices[-1], 1] == segments[:, 0])

    return segments[new_indices]


class Mesh(object):
    """
    A 2D mesh object, representing a single contiguous geometrical object
    """
    def __init__(self, segments):
        self.segments = segments
    
    def __add__(self, other):
        return Mesh(np.concatenate([self.segments, other.segments]))

    def __iter__(self):
        return iter(self.segments)

    def translate(self, shift):
        """
        Translate the 2D mesh by `shift` amount

        Parameters
        ----------
        shift : (2) ndarray
            [shift_x, shift_y]
        """
        self.segments[:, :, 0] += shift[0]
        self.segments[:, :, 1] += shift[1]

    def rotate(self, angle, pivot=[0., 0.], degrees=True):
        """
        Rotate the 2D mesh by `angle` degrees about a point `pivot`

        Parameters
        ----------
        angle : float
            Amount to rotate in degrees (default)
        pivot : (2) ndarray
            Point about which to pivot, [pivot_x, pivot_y]
        degrees : bool
            Whether angle is in degrees (default) or in radians
        """
        self.segments[:, :, 0] -= pivot[0]
        self.segments[:, :, 1] -= pivot[1]

        if degrees:
            angle = np.deg2rad(angle)

        new_xs = self.segments[:, :, 0] * np.cos(angle) - self.segments[:, :, 1] * np.sin(angle)
        new_ys = self.segments[:, :, 0] * np.sin(angle) + self.segments[:, :, 1] * np.cos(angle)

        self.segments[:, :, 0] = new_xs - pivot[0]
        self.segments[:, :, 1] = new_ys - pivot[1]


def create_rectangle(width, height):
    """
    Create a rectangular mesh

    Parameters
    ----------
    width : float
    height : float

    Returns
    -------
    out : Mesh

    """
    points = np.zeros((5, 2), dtype=np.float32)

    points[0] = [width/2., height/2.]
    points[1] = [width/2., -height/2.]
    points[2] = [-width/2., -height/2.]
    points[3] = [-width/2., height/2.]
    points[4] = [width/2., height/2.]
    
    segments = math2d.create_segments_from_points(points)
    
    return Mesh(segments)


def create_circle(radius, n_segments=20):
    """
    Create a circular mesh

    Parameters
    ----------
    radius : float
    n_segments : int

    Returns
    -------
    out : Mesh

    """
    points = np.zeros((n_segments + 1, 2), dtype=np.float32)
    
    radians = np.linspace(0., 2 * np.pi, n_segments + 1)[:-1][::-1]
    points[:-1, 0] = np.cos(radians) * radius
    points[:-1, 1] = np.sin(radians) * radius

    points[-1] = points[0]
    
    segments = math2d.create_segments_from_points(points)
    
    return Mesh(segments)


def create_hollow(outer_object, inner_object):
    """
    Create a combined hollow mesh

    Parameters
    ----------
    outer_object, inner_object : Mesh

    Returns
    -------
    out : Mesh
        Combined mesh which is hollow as defined by outer_object and inner_object

    """
    inner_object.segments = np.fliplr(inner_object.segments)[::-1]

    return outer_object + inner_object
