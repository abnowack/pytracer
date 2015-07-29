import numpy as np

def angle_matrix(angle, radian=False):
    """Generate 2D rotation matrix."""
    if not radian:
        angle = angle / 180. * np.pi
    
    return np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

def translate_rotate_mesh(meshes, translate=np.zeros((2)), rotate=np.identity(2)):
    """Translate and rotate mesh or list of meshes."""
    try:
        iterator = iter(meshes)
    except TypeError:
        meshes.points = np.inner(meshes.points, rotate) + translate
    else:
        for mesh in meshes:
            mesh.points = np.inner(mesh.points, rotate) + translate

class Mesh(object):
    """Wrapper around list of points and lixels which refer to points."""
    def __init__(self, points, lixels):
        """Points (N,2) floats, lixels (N,2) ints."""
        self.points = points
        self.lixels = lixels
    
    def __add__(self, other):
        return Mesh(np.concatenate([self.points, other.points]),
                    np.concatenate([self.lixels, other.lixels + np.size(self.lixels, 0)]))

    def lixel_normal(self, i):
        """
        Compute normal defined by (L[0] - L[1]) cross (0, 0, 1).

        (L[1] - L[0]) is vector from L[1] to L[0]
        """
        lixel = self.lixels[i]
        
        points = self.points[lixel] 
        # L[0].y - L[1].y
        px = points[0, 1] - points[1, 1]
        # L[1].x - L[0].x
        py = points[1, 0] - points[0, 0]
        length = np.sqrt(px ** 2. + py ** 2.)
        return np.array([px / length, py / length], dtype=np.float32)
    
    def continuous_path_order(self):
        """
        mesh points not neccessarily in order, reorganize points such that
        point[i], point[i+1], ... point[0] will trace out continuous closed path
        """
        new_index = []
    
        next_index = 0
    
        for i, lixel in enumerate(self.lixels):
            if next_index not in new_index:
                new_index.append(next_index)
            else:
                others = np.setdiff1d(self.lixels, self.lixels[new_index])
                next_index = np.where(self.lixels[others[0]] == self.lixels)[0][0]
                new_index.append(next_index)
    
            next_index = np.where(self.lixels[new_index[-1], 1] == self.lixels[:, 0])[0][0]
    
        return self.lixels[new_index], self.points[new_index]

def create_rectangle(w, h):
    points = np.zeros((4, 2), dtype=np.float32)
    lixels = np.zeros((4, 2), dtype=np.int32)
    
    points[:, 1] = h / 2.
    points[:, 0] = w / 2.
    points[2, :] *= -1
    points[1, 0] *= -1
    points[3, 1] *= -1
    
    lixels[:, 0] = np.arange(np.size(points, 0))
    lixels[:, 1] = np.roll(lixels[:, 0], 1)
    
    return Mesh(points, lixels)

def create_circle(radius, n_segments=20):
    points = np.zeros((n_segments, 2), dtype=np.float32)
    lixels = np.zeros((n_segments, 2), dtype=np.int32)
    
    radians = np.linspace(0., 2 * np.pi, n_segments + 1)[:-1]
    points[:, 0] = np.cos(radians) * radius
    points[:, 1] = np.sin(radians) * radius
    
    lixels[:, 0] = np.arange(np.size(points, 0))
    lixels[:, 1] = np.roll(lixels[:, 0], 1)
    
    return Mesh(points, lixels)

def create_hollow(outer_object, inner_object):
    """ Must respect outer and inner object argument order. """
    inner_object.points = inner_object.points[::-1]

    return outer_object + inner_object