import numpy as np


def normal(segment):
    """
    Create a normal vector from a 2d line segment

    Parameters
    ----------
    segment : (2, 2) array_like
        [[point1_x, point1_y], [point2_x, point2_y]]

    Returns
    -------
    out : (2, 2) ndarray
        Normalized vector which begins at the midpoint of the segment and is perpendicular to the segment
        [[point1_x, point1_y], [point2_x, point2_y]]

    """
    dx = segment[0, 1] - segment[1, 1]
    dy = segment[1, 0] - segment[0, 0]
    length = np.sqrt(dx ** 2. + dy ** 2.)
    return np.array([dx / length, dy / length], dtype=np.float32)


def solid_angle(segment, point):
    """
    Returns the 2D solid angle of a segment relative to a point

    Parameters
    ----------
    segment : (2, 2) array_like
        [[point1_x, point1_y], [point2_x, point2_y]]
    point : (2) array_like
        [point_x, point_y]

    Returns
    -------
    angle : float
        2D solid angle

    """
    X = segment[0] - point
    Y = segment[1] - point
    X_Y = X - Y
    cos_angle = (np.dot(X, X) + np.dot(Y, Y) - np.dot(X - Y, X - Y)) / (2 * np.linalg.norm(X) * np.linalg.norm(Y))
    angle = np.arccos(cos_angle)
    if angle > np.pi / 2.:
        angle = np.pi - angle
    return angle


def intersect(segment, other_segment, other_is_ray=False):
    """
    Calculate the point of intersection between two line segments, or a line segment and ray

    Parameters
    ----------
    segment, other_segment : (2, 2) array_like
        [[point1_x, point1_y], [point2_x, point2_y]]
    other_is_ray : bool
        other_segment represents a ray starting at point_1 and extending for infinity

    Returns
    -------
    out : (2) float or None
        Point of intersection, or None if intersection didn't occur

    """
    epsilon = 1e-15
    p, q = segment[0], other_segment[0]
    r, s = segment[1] - segment[0], other_segment[1] - other_segment[0]
    
    denom = r[0] * s[1] - r[1] * s[0]

    # colinear or parallel
    if denom == 0.:
        return None

    u_num = (q - p)[0] * r[1] - (q - p)[1] * r[0]
    t_num = (q - p)[0] * s[1] - (q - p)[1] * s[0]
    t, u = t_num / denom, u_num / denom
    intersection = p + t * r

    # contained with both line segments
    # must shift over line segment by epsilon to prevent double overlapping
    if -epsilon < t < 1. - epsilon:
        if not ray or 0. < u <= 1.:
            return intersection


def angle_matrix(angle, radian=False):
    """
    Generate 2D rotation matrix.

    Parameters
    ----------
    angle : float
        2D Rotation Angle
    radian : bool
        Whether angle is specified as degrees (default) or in radians

    Returns
    -------
    rot_matrix : (2, 2) ndarray
        2D rotation transformation matrix
    """
    if not radian:
        angle = angle / 180. * np.pi

    rot_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    
    return rot_matrix


def create_segments_from_points(points):
    """
    Convert contiguously ordered points to an array of line segments

    Parameters
    ----------
    points : (M, 2) array_like
        Array of points which are ordered to trace out a contiguous path

    Returns
    -------
    segments : (M, 2, 2) ndarray
        Array of line segments where each index is [[point1_x, point1_y], [point2_x, point2_y]]

    """
    segments = np.zeros((np.size(points, 0) - 1, 2, 2))
    segments[:, :, 0][:, 0] = points[:, 0][:-1]
    segments[:, :, 0][:, 1] = points[:, 0][1:]
    segments[:, :, 1][:, 0] = points[:, 1][:-1]
    segments[:, :, 1][:, 1] = points[:, 1][1:]
    return segments
