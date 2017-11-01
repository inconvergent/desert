# -*- coding: utf-8 -*-

from numpy import reshape
from numpy import float32 as npfloat


class Rgba():
  def __init__(self, c):
    r, g, b, a = c
    self.rgba = reshape([a*r , a*g, a*b, a], (1, 4)).astype(npfloat)

  def __repr__(self):
    a = self.rgba[0, 3]
    return '<Rgba {:0.3f} {:0.3f} {:0.3f} {:0.3f}>'.format(
        self.rgba[0, 0]/a,
        self.rgba[0, 1]/a,
        self.rgba[0, 2]/a,
        a)

