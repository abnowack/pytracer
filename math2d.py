import numpy as np

def normal(segment):
    dx = segment[0, 1] - segment[1, 1]
    dy = segment[1, 0] - segment[0, 0]
    length = np.sqrt(dx ** 2. + dy ** 2.)
    return np.array([dx / length, dy / length], dtype=np.float32)

def solid_angle(segment, point):
    X = segment[0] - point
    Y = segment[1] - point
    X_Y = X - Y
    cos_angle = (np.dot(X, X) + np.dot(Y, Y) - np.dot(X - Y, X - Y)) / (2 * np.linalg.norm(X) * np.linalg.norm(Y))
    angle = np.arccos(cos_angle)
    if angle > np.pi / 2.:
        angle = np.pi - angle
    return angle

def intersect(segment, other_segment, ray=False):
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

def center(segment):
    return (segment[0] + segment[1]) / 2.

def angle_matrix(angle, radian=False):
    """Generate 2D rotation matrix."""
    if not radian:
        angle = angle / 180. * np.pi
    
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

def create_segments_from_points(points):
    ''' Assumes points are ordered and trace out a contiguous path '''
    segments = np.zeros((np.size(points, 0) - 1, 2, 2))
    segments[:, :, 0][:, 0] = points[:, 0][:-1]
    segments[:, :, 0][:, 1] = points[:, 0][1:]
    segments[:, :, 1][:, 0] = points[:, 1][:-1]
    segments[:, :, 1][:, 1] = points[:, 1][1:]
    return segments