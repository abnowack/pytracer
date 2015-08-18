import numpy as np
import math2d

def continuous_path_order(segments):
    ordered_segments = np.zeros_like(segments)
    indices = range(np.size(segments, 0))
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
    """Wrapper around list of points and lixels which refer to points."""
    def __init__(self, segments):
        """ Segments """
        self.segments = segments
    
    def __add__(self, other):
        return Mesh(np.concatenate([self.segments, other.segments]))

    def __iter__(self):
        return iter(self.segments)

    def translate(self, shift):
        self.segments[:, :, 0] += shift[0]
        self.segments[:, :, 1] += shift[1]

    def rotate(self, angle, pivot=[0., 0.], degrees=True):
        self.segments[:, :, 0] -= pivot[0]
        self.segments[:, :, 1] -= pivot[1]

        if degrees:
            angle = np.deg2rad(angle)

        new_xs = self.segments[:, :, 0] * np.cos(angle) - self.segments[:, :, 1] * np.sin(angle)
        new_ys = self.segments[:, :, 0] * np.sin(angle) + self.segments[:, :, 1] * np.cos(angle)

        self.segments[:, :, 0] = new_xs - pivot[0]
        self.segments[:, :, 1] = new_ys - pivot[1]

def create_rectangle(w, h):
    points = np.zeros((5, 2), dtype=np.float32)

    points[0] = [w , h]
    points[1] = [w, -h]
    points[2] = [-w, -h]
    points[3] = [-w, h]
    points[4] = [w, h]
    points /= 2.
    
    segments = math2d.create_segments_from_points(points)
    
    return Mesh(segments)

def create_circle(radius, n_segments=20):
    points = np.zeros((n_segments + 1, 2), dtype=np.float32)
    
    radians = np.linspace(0., 2 * np.pi, n_segments + 1)[:-1][::-1]
    points[:-1, 0] = np.cos(radians) * radius
    points[:-1, 1] = np.sin(radians) * radius

    points[-1] = points[0]
    
    segments = math2d.create_segments_from_points(points)
    
    return Mesh(segments)

def create_hollow(outer_object, inner_object):
    """ Must respect outer and inner object argument order. """
    inner_object.segments = np.fliplr(inner_object.segments)[::-1]

    return outer_object + inner_object