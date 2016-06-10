import numpy as np


def center(segments):
    """
    Calculate center (midpoint) of a line segment

    Parameters
    ----------
    segment : (2, 2) array_like
        [[point1_x, point1_y], [point2_x, point2_y]]

    Returns
    -------
    seg_center : (2) array_like
        [center_x, center_y]
    """
    if segments.ndim > 2:
        sp1 = segments[:, 0]
        sp2 = segments[:, 1]
    else:
        sp1 = segments[0]
        sp2 = segments[1]

    seg_center = (sp1 + sp2) / 2.
    return seg_center


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


def solid_angle(segments, point):
    """
    Returns the 2D solid angle of a segment relative to a point

    Parameters
    ----------
    segments : (, 2, 2) ndarray
        [[point1_x, point1_y], [point2_x, point2_y]]
    point : (2) ndarray
        [point_x, point_y]

    Returns
    -------
    angle : float
        2D solid angle

    Notes
    -----
    Calculates using law of cosines where the solid angle is defined in terms of the lengths of sides A, B, C

    angle = arccos((A*A + B*B - C*C) / (2 * A * B))

    Where
    A = |sp1 - point|
    B = |sp2 - point|
    C = |sp1 - sp2|

    """
    if segments.ndim > 2:
        sp1 = segments[:, 0]
        sp2 = segments[:, 1]

    t = sp1 - point

    a = np.linalg.norm(sp1 - point, axis=1)
    b = np.linalg.norm(sp2 - point, axis=1)
    c = np.linalg.norm(sp1 - sp2, axis=1)

    num = np.power(a, 2) + np.power(b, 2) - np.power(c, 2)
    denom = 2 * a * b

    angle = np.arccos(np.abs(num / denom))
    angle[angle > np.pi / 2.] = np.pi - angle[angle > np.pi / 2.]

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
        if not other_is_ray or 0. < u <= 1.:
            return intersection

# Still slower than the C implementation!!
def intersects(segments, other_segment, other_is_ray=False):
    epsilon = 1e-15
    p = segments[:, 0]
    q = other_segment[0]
    r = segments[:, 1] - segments[:, 0]
    s = other_segment[1] - other_segment[0]

    denom = r[:, 0] * s[1] - r[:, 1] * s[0]

    denom_mask = denom != 0.

    u_num = (q - p)[:, 0] * r[:, 1] - (q - p)[:, 1] * r[:, 0]
    t_num = (q - p)[:, 0] * s[1] - (q - p)[:, 1] * s[0]

    t = t_num / denom
    u = u_num / denom

    intersection = t[:, np.newaxis] * r
    intersection += p

    t_mask = (-epsilon < t) & (t < 1. - epsilon)
    mask = denom_mask & t_mask

    if not other_is_ray:
        u_mask = (0. < u) & (u <= 1.)
        mask = u_mask & mask

    indexes,  = np.where(mask)
    return intersection[mask], indexes

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
