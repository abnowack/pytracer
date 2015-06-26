# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:01:55 2015

@author: Aaron
"""

class Material(object):
    def __init__(self, attenuation, color='black'):
        self.attenuation = attenuation
        self.color = color
    
    def __eq__(self, other):
        return self.attenuation == other.attenuation