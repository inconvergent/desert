# -*- coding: utf-8 -*-

from numpy import float32 as npfloat
from numpy import int32 as npint
from numpy import reshape
from numpy import prod


class Box():
  def __init__(self, s, mid, dens):

    try:
      sx, sy = s
    except TypeError:
      sx = s
      sy = s

    self.s = reshape([sx, sy], (1, 2)).astype(npfloat)
    self.mid = reshape(mid, (1, 2)).astype(npfloat)
    self.dens = dens

  def __repr__(self):
    return '<Box: (s: ({:f} {:f}) xy ({:f}, {:f}) d: {:f})>'.format(
        self.s[0,0], self.s[0, 1],
        self.mid[0, 0], self.mid[0, 1],
        self.dens)

  def get_n(self, imsize):
    s = self.s
    return int(4*prod(s, axis=1)*self.dens*(imsize**2))

