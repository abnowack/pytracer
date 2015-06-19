# -*- coding: utf-8 -*-
"""
Created on Fri Jun 05 11:50:01 2015

@author: Aaron

TODO: Implement Geometry Checking
    - Test if any lixels overlap
    - Cannot have hole in a hole, or solid in a solid
TODO: Use Bounded Volume Heirarchy to Reduce Lixel Search
TODO: Remove out of bounds Lixels
TODO: Material Renderer
"""
import numpy as np

class Point(object):
    def __init__(self, x, y):
        self.__x__, self.__y__ = x, y
    
    def __str__(self):
        return '({0}, {1})'.format(self.x, self.y)
    
    @property
    def x(self):
        return self.__x__

    @property    
    def y(self):
        return self.__y__
    
    def __eq__(self, other):
        return self.__x__ == other.__x__ and self.__y__ == self.__y__
    
    def __add__(self, other):
        return Point(self.__x__ + other.__x__, self.__y__ + other.__y__)
    
    def __radd__(self, other):
        return Point.__add__(self, other)
    
    def __sub__(self, other):
        return Point(self.__x__ - other.__x__, self.__y__ - other.__y__)
    
    def __rsub__(self, other):
        return Point.__sub__(self, other)
    
    def __mul__(self, other):
        return Point(self.__x__ * other, self.__y__ * other)
    
    def __rmul__(self, other):
        return Point.__mul__(self, other)
    
    def __div__(self, other):
        return Point.__mul__(self, 1./other)
    
    def __rdiv__(self, other):
        return Point.__div__(self, other)

class Lixel(object):
    def __init__(self, point_1, point_2):
        self.points = [point_1, point_2]
    
    def normal(self):
        """ Vector defined by cross-product of (0,0,1) and P2-P1 """
        px = self.points[0].y - self.points[1].y
        py = self.points[1].x - self.points[0].x
        length = np.sqrt(px**2. + py**2.)
        return Point(px / length, py / length)
    
    def center(self):
        center_x = (self.points[0].x + self.points[1].x) / 2.
        center_y = (self.points[0].y + self.points[1].y) / 2.
        return Point(center_x, center_y)
    
    def calc_intercept(self, point_1, point_2):
        ''' Given Line Segments:
            - p, p+r
            - q, q+s
            Find intercept point where: p + t * r = q + u * s
            where 0 <= t <= 1, 0 <= u <= 1
            
            t = (q - p) x s / (r x s)
            u = (q - p) x r / (r x s)
            
            where a x b = a_x * b_y - a_y * b_x
        '''
        
        p = point_1
        r = point_2 - point_1
        q = self.points[0]
        s = self.points[1] - self.points[0]
        
        denom = r.x * s.y - r.y * s.x
        u_num = (q - p).x * r.y - (q - p).y * r.x
        t_num = (q - p).x * s.y - (q - p).y * s.x
        
        if denom == 0. and u_num == 0.:
            # colinear
            return None # TODO: what should this really return?
        elif denom == 0. and u_num != 0.:
            # parallel
            return None
            
        t = t_num / denom
        u = u_num / denom

        if 0 <= t <= 1. and 0 <= u <= 1.:
            return p + t * r
        else:
            # beyond line segment boundary
            return None

class Geometry(object):
    def __init__(self):
        self.__attenuation__ = 0.
    
    @property
    def attenuation(self):
        return self.__attenuation__
    
    """ Points must be given in clockwise orientation """
    def generate_points(self, *args):
        return []
    
    def Lixelate(self, n_lixels=4, positive=True):
        points = self.generate_points(n_lixels, positive)
        
        lixels = []
        for i in xrange(len(points)):
            p1 = points[i]
            p2 = points[(i+1) % len(points)]
            
            lixel = Lixel(p1, p2)
            lixels.append(lixel)
        return lixels

class Rectangle(Geometry):
    def __init__(self, point_1, point_2, attenuation=0.):
        self.__point_1__, self.__point_2__ = point_1, point_2
        self.__attenuation__ = attenuation
    
    def generate_points(self, n_lidels=4, positive=True):
        center_x = (self.__point_1__.x + self.__point_2__.x) / 2.
        center_y = (self.__point_1__.y + self.__point_2__.y) / 2.
        width = abs(self.__point_1__.x - self.__point_2__.x)
        height = abs(self.__point_1__.y - self.__point_2__.y)
        
        point_1 = Point(center_x - width / 2., center_y - height / 2.)
        point_2 = Point(center_x - width / 2., center_y + height / 2.)
        point_3 = Point(center_x + width / 2., center_y + height / 2.)
        point_4 = Point(center_x + width / 2., center_y - height / 2.)
        
        points = [point_1, point_2, point_3, point_4]
        
        if positive:
            return points
        else:
            return points[::-1]

class Circle(Geometry):
    def __init__(self, center, radius, attenuation=0.):
        self.__center__, self.__radius__ = center, radius
        self.__attenuation__ = attenuation
    
    def generate_points(self, n_lixels=20, positive=True):
        # TODO: adjust radius to be aligned for approximate circle
        points = []
        radians = np.linspace(2. * np.pi, 0., n_lixels+1)[0:-1]
        for radian in radians:
            point_x = self.__center__.x + self.__radius__ * np.cos(radian)
            point_y = self.__center__.y + self.__radius__ * np.sin(radian)
            points.append(Point(point_x, point_y))
        if positive:
            return points
        else:
            return points[::-1]

class Torus(object):
    def __init__(self, center, inner_radius, outer_radius, attenuation=0.):
        self.__center__ = center
        self.__inner_radius__ = inner_radius
        self.__outer_radius__ = outer_radius
        self.__attenuation__ = attenuation
        
        self.__inner_circle__ = Circle(center, inner_radius, 0.)
        self.__outer_circle__ = Circle(center, outer_radius, attenuation)
    
    def Lixelate(self, n_lixels=20):
        outer_lixels = self.__outer_circle__.Lixelate(n_lixels)
        inner_lixels = self.__inner_circle__.Lixelate(n_lixels, False)
        return outer_lixels + inner_lixels

class HollowObject(object):
    ''' Duck typed class for composition of objects with hollow center '''
    def __init__(self, outer_object, inner_object, attenuation):
        self.__outer_object__ = outer_object
        self.__inner_object__ = inner_object
        self.__attenuation__ = attenuation
    
    @property
    def attenuation(self):
        return self.__attenuation__
    
    def Lixelate(self, n_lixels=20):
        outer_lixels = self.__outer_object__.Lixelate(n_lixels)
        inner_lixels = self.__inner_object__.Lixelate(n_lixels, False)
        return outer_lixels + inner_lixels

class DetectorPlane(object):
    def __init__(self, center, angle, length, nbins):
        self.__center__ = center
        self.__angle__ = angle
        self.__length__ = length
        self.__nbins__ = nbins
        
        self.__bins__ = self.__generate_bins__()
    
    def __generate_bins__(self):
        bins = np.zeros((self.__nbins__,2))
        bins[:, 1] = np.linspace(-self.__length__/2., self.__length__/2., self.__nbins__)
        
        rot_matrix = np.array([[np.cos(self.__angle__), -np.sin(self.__angle__)],
                               [np.sin(self.__angle__), np.cos(self.__angle__)]])
        
        bins = np.dot(bins, rot_matrix)
        bins[:, 0] += self.__center__.x
        bins[:, 1] += self.__center__.y
        
        return bins
        
def draw_lixels(lixels, draw_normals=True):
    ax = plt.axes()
    
    for lixel in lixels:
        plt.plot([p.x for p in lixel.points], [p.y for p in lixel.points])
        
        if draw_normals:
            center = lixel.center()
            normal = lixel.normal()
            ax.arrow(center.x, center.y, normal.x, normal.y)
        
def draw_detector_plane(detector):
    plt.plot(detector.__bins__[:, 0], detector.__bins__[:, 1], marker='+', ms=10.)

def draw_everything(lixels, detector, source, draw_normals=True):
    draw_lixels(lixels, draw_normals)
    draw_detector_plane(detector)
    plt.scatter(source.x, source.y, c='r')
    plt.axis('equal')
    plt.show()

def detector_geometry_distance(geolixels, source, detector):
    distances = np.zeros(np.size(detector.__bins__, 0) - 1)
    
    for i in xrange(len(distances)):
        lixel_distance_buffer = []
        bin_edge_0 = Point(detector.__bins__[i, 0], detector.__bins__[i, 1])
        bin_edge_1 = Point(detector.__bins__[i+1, 0], detector.__bins__[i+1, 1])
        center = (bin_edge_0 + bin_edge_1) / 2.
        for lixel in geolixels:
            intercept = lixel.calc_intercept(source, center)
            if intercept is not None:
                distance = np.sqrt((source.x - intercept.x)**2. + (source.y - intercept.y)**2.)
                sign = (source - intercept).x * lixel.normal().x + \
                       (source - intercept).y * lixel.normal().y
                if sign < 0:
                    distances[i] += distance
                else:
                    distances[i] -= distance
        
    return distances

def multiple_geometries(geometries, source, detector, detail=50):
    total_attenuation = None
    for geo in geometries:
        geo_pathlength = detector_geometry_distance(geo.Lixelate(detail), source, detector)
        attenuation = geo.attenuation * geo_pathlength
        if total_attenuation is None:
            total_attenuation = attenuation
        else:
            total_attenuation += attenuation
    
    return total_attenuation
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Assembly
    box = HollowObject(Rectangle(Point(40, 10), Point(70, -10)), 
                       Rectangle(Point(42,  8), Point(68,  -8)), 0.5)
    casting = HollowObject(Circle(Point(50, 0), 7), Circle(Point(50, 0), 5), 0.8)
    small_box_1 = Rectangle(Point(62, 6), Point(66, 2), 0.2)
    small_box_2 = Rectangle(Point(62, -2), Point(66, -6), 0.4)
    
    lixels = box.Lixelate()
    lixels.extend(casting.Lixelate(50))
    lixels.extend(small_box_1.Lixelate())
    lixels.extend(small_box_2.Lixelate())
    
    plt.figure()

    n_angles = 1
    n_bins = 1000
    sinogram = np.zeros((n_bins, n_angles))    
    
    for i, angle in enumerate(np.linspace(0, 2. * np.pi, n_angles+1)[:-1]):
        print i
        source = Point(50.-np.cos(angle)*50, np.sin(angle)*50.)
        print source.x, source.y
        detector = DetectorPlane(Point(np.cos(angle)*50.+50, -np.sin(angle)*50), angle, 100, n_bins+1)
    
        plt.figure()
        draw_everything(lixels, detector, source, False)
    
#        plt.figure()
        distances = detector_geometry_distance(lixels, source, detector)    
        attenuation = multiple_geometries([box, casting, small_box_1, small_box_2],
                                          source, detector)
        sinogram[:, i] = attenuation
#        plt.plot(attenuation)