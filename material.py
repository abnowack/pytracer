# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:01:55 2015

@author: Aaron
"""

class Material(object):
    def __init__(self, attenuation, macro_fission=0, color='black'):
        self.attenuation = attenuation
        self.macro_fission = macro_fission
        self.color = color
    
    def __eq__(self, other):
        return self.attenuation == other.attenuation and self.macro_fission == other.macro_fission

    @property
    def is_fissionable(self):
        return self.macro_fission > 0.0