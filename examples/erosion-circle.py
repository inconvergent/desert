#!/usr/bin/python3
# -*- coding: utf-8 -*-


from sys import argv

from desert.helpers import filename
from numpy import cos
from numpy import sin
from numpy import zeros
from numpy.random import random

from erosion import Erosion
from desert import Desert
from desert import stroke

from desert.color import rgb
from desert.color import white
from desert.color import black

from time import time


VERBOSE = 'v'

def main(arg):

  with Desert(1000, show=True, verbose=VERBOSE)\
      .init(fg=black(0.001),
            bg=white()) as c:

    a = random(2)
    acc = zeros(2)
    noise = 0.00000001
    rad = 0.45

    resa = []
    resb = []

    for i in range(2000000):
      a += acc
      acc += (1.0-2*random(2))*noise

      x1 = 0.5 + cos(a[0])*rad
      y1 = 0.5 + sin(a[0])*rad
      x2 = 0.5 + cos(a[1])*rad
      y2 = 0.5 + sin(a[1])*rad

      resa.append((x1, y1))
      resb.append((x2, y2))

      if not i%50000:
        c.gdraw([stroke(resa, resb, 0.02)])
        resa = []
        resb = []
        c.show()

    c.save(filename(arg), gamma=1.5)


if __name__ == '__main__':
  main(argv)

