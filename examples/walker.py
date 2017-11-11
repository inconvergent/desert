#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sys import argv

from numpy.random import random
from numpy import row_stack

from desert import Desert
from desert import box
from desert import stroke

from desert.color import black
from desert.color import white

from desert.rnd import in_circle

from desert.helpers import filename


VERBOSE = 'vv'


def main(arg):

  imsize = 1000
  with Desert(imsize, verbose=VERBOSE)\
      .init(fg=black(0.1),
            bg=white()) as desert:

    num = 20

    xya = random((num, 2))
    xyb = random((num, 2))

    stacka = []
    stackb = []

    drift = in_circle(1, 0, 0, 0.00001)

    for i in range(1, 100000):
      xya += in_circle(num, 0, 0, 0.001) + drift
      xyb += in_circle(num, 0, 0, 0.001) + drift
      stacka.append(xya.copy())
      stackb.append(xyb.copy())

      if not i%10000:
        desert.draw([stroke(stacka, stackb, 0.01)]).show(0.01)
        stacka = []
        stackb = []

    desert.show(1).save(filename(arg))


if __name__ == '__main__':
  main(argv)

