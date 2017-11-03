# -*- coding: utf-8 -*-

from numpy import reshape
from numpy import float32 as npfloat


class rgba():
  def __init__(self, r, g, b, a=1.0):
    self.rgba = reshape([a*r , a*g, a*b, a], (1, 4)).astype(npfloat)

  def __repr__(self):
    a = self.rgba[0, 3]
    return '<rgba {:0.3f} {:0.3f} {:0.3f} {:0.3f}>'.format(
        self.rgba[0, 0]/a,
        self.rgba[0, 1]/a,
        self.rgba[0, 2]/a,
        a)


def rgb(r, g, b, a=1.0):
  return rgba(r, g, b, a)


def white(a=1.0):
  return rgba(1, 1, 1, a)


def black(a=1.0):
  return rgba(0, 0, 0, a)

